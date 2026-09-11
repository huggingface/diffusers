# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from ...utils import logging
from .core import Conversion, Rule
from .transforms import ReorderChunks, Split


logger = logging.get_logger(__name__)

COGVIDEOX_TRANSFORMER_PREFIX = "model.diffusion_model."


def cogvideox_fixed_position_embedding(config, dtype):
    """Recreate the non-persistent Diffusers position buffer saved by the original SAT runtime."""
    from ...models.embeddings import get_3d_sincos_pos_embed

    hidden_size = config["num_attention_heads"] * config["attention_head_dim"]
    width = config["sample_width"] // config["patch_size"]
    height = config["sample_height"] // config["patch_size"]
    frames = (config["sample_frames"] - 1) // config["temporal_compression_ratio"] + 1
    positions = get_3d_sincos_pos_embed(
        hidden_size,
        (width, height),
        frames,
        config["spatial_interpolation_scale"],
        config["temporal_interpolation_scale"],
        output_type="pt",
    ).flatten(0, 1)
    text_positions = positions.new_zeros(config["max_text_seq_length"], hidden_size)
    return torch.cat((text_positions, positions)).unsqueeze(0).to(dtype=dtype)


@dataclass(frozen=True)
class CogVideoXAdaLN:
    """Separate SAT's interleaved attention/MLP modulation into the two Diffusers norms."""

    hidden_size: int

    def forward(self, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        tensors = ReorderChunks((0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11)).forward(tensors)
        return Split((6 * self.hidden_size, 6 * self.hidden_size)).forward(tensors)

    def inverse(self, tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        tensors = Split((6 * self.hidden_size, 6 * self.hidden_size)).inverse(tensors)
        return ReorderChunks((0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11)).inverse(tensors)


def cogvideox_transformer_conversion(config: Mapping[str, Any]) -> Conversion:
    """Build SAT <-> Diffusers transformer weight rules from a `CogVideoXTransformer3DModel.config`.

    The original keys are relative to `model.diffusion_model.`. Non-learned positional state and training-only
    checkpoint entries are outside this component conversion. No prior import or model allocation is required.
    """
    hidden_size = config["num_attention_heads"] * config["attention_head_dim"]
    modules = [
        ("time_embed.0", "time_embedding.linear_1"),
        ("time_embed.2", "time_embedding.linear_2"),
        ("mixins.patch_embed.text_proj", "patch_embed.text_proj"),
        ("mixins.final_layer.linear", "proj_out"),
        ("mixins.final_layer.adaLN_modulation.1", "norm_out.linear"),
    ]
    if config.get("ofs_embed_dim") is not None:
        modules.extend([("ofs_embed.0", "ofs_embedding.linear_1"), ("ofs_embed.2", "ofs_embedding.linear_2")])
    if config.get("norm_elementwise_affine", True):
        modules.extend(
            [("transformer.final_layernorm", "norm_final"), ("mixins.final_layer.norm_final", "norm_out.norm")]
        )

    mapping = {"mixins.patch_embed.proj.weight": "patch_embed.proj.weight"}
    # The 1.5 linear patch embed has a bias even when patch_bias=False (that flag configures the 1.0 convolution).
    if config.get("patch_bias", True) or config.get("patch_size_t") is not None:
        mapping["mixins.patch_embed.proj.bias"] = "patch_embed.proj.bias"
    if config.get("use_learned_positional_embeddings", False):
        mapping["mixins.pos_embed.pos_embedding"] = "patch_embed.pos_embedding"

    rules = []
    for i in range(config["num_layers"]):
        original = f"transformer.layers.{i}"
        diffusers = f"transformer_blocks.{i}"
        modules.extend(
            [
                (f"{original}.attention.dense", f"{diffusers}.attn1.to_out.0"),
                (f"{original}.mlp.dense_h_to_4h", f"{diffusers}.ff.net.0.proj"),
                (f"{original}.mlp.dense_4h_to_h", f"{diffusers}.ff.net.2"),
                (f"mixins.adaln_layer.query_layernorm_list.{i}", f"{diffusers}.attn1.norm_q"),
                (f"mixins.adaln_layer.key_layernorm_list.{i}", f"{diffusers}.attn1.norm_k"),
            ]
        )
        if config.get("norm_elementwise_affine", True):
            modules.extend(
                [
                    (f"{original}.input_layernorm", f"{diffusers}.norm1.norm"),
                    (f"{original}.post_attention_layernorm", f"{diffusers}.norm2.norm"),
                ]
            )
        for parameter in ("weight", "bias"):
            if parameter == "weight" or config.get("attention_bias", True):
                rules.append(
                    Rule(
                        (f"{original}.attention.query_key_value.{parameter}",),
                        tuple(f"{diffusers}.attn1.to_{name}.{parameter}" for name in "qkv"),
                        Split((hidden_size, hidden_size, hidden_size)),
                    )
                )
            rules.append(
                Rule(
                    (f"mixins.adaln_layer.adaLN_modulations.{i}.1.{parameter}",),
                    (f"{diffusers}.norm1.linear.{parameter}", f"{diffusers}.norm2.linear.{parameter}"),
                    CogVideoXAdaLN(hidden_size),
                )
            )

    mapping.update({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})
    return Conversion(mapping, tuple(rules))


def cogvideox_vae_conversion(config: Mapping[str, Any]) -> Conversion:
    """Build SAT <-> Diffusers VAE weight rules from an `AutoencoderKLCogVideoX.config`."""
    channels = config["block_out_channels"]
    layers = config["layers_per_block"]
    modules = []
    resnets = []
    for component in ("encoder", "decoder"):
        modules.extend((f"{component}.{name}", f"{component}.{name}") for name in ("conv_in.conv", "conv_out.conv"))
        for i in range(2):
            resnets.append(
                (f"{component}.mid.block_{i + 1}", f"{component}.mid_block.resnets.{i}", channels[-1], channels[-1])
            )
        block_channels = channels if component == "encoder" else tuple(reversed(channels))
        previous_channels = block_channels[0]
        for i, out_channels in enumerate(block_channels):
            if component == "encoder":
                original, diffusers = f"encoder.down.{i}", f"encoder.down_blocks.{i}"
                sample_original, sample_diffusers = "downsample", "downsamplers.0"
            else:
                original, diffusers = f"decoder.up.{len(channels) - 1 - i}", f"decoder.up_blocks.{i}"
                sample_original, sample_diffusers = "upsample", "upsamplers.0"
            for j in range(layers + (component == "decoder")):
                resnets.append((f"{original}.block.{j}", f"{diffusers}.resnets.{j}", previous_channels, out_channels))
                previous_channels = out_channels
            if i < len(channels) - 1:
                modules.append((f"{original}.{sample_original}.conv", f"{diffusers}.{sample_diffusers}.conv"))

    for original, diffusers, in_channels, out_channels in resnets:
        modules.extend((f"{original}.{name}", f"{diffusers}.{name}") for name in ("conv1.conv", "conv2.conv"))
        norm_names = ("norm1", "norm2")
        if original.startswith("decoder."):
            norm_names = tuple(
                f"{norm}.{name}" for norm in norm_names for name in ("norm_layer", "conv_y.conv", "conv_b.conv")
            )
        modules.extend((f"{original}.{name}", f"{diffusers}.{name}") for name in norm_names)
        if in_channels != out_channels:
            modules.append((f"{original}.nin_shortcut", f"{diffusers}.conv_shortcut"))

    modules.append(("encoder.norm_out", "encoder.norm_out"))
    modules.extend(
        (f"decoder.norm_out.{name}", f"decoder.norm_out.{name}")
        for name in ("norm_layer", "conv_y.conv", "conv_b.conv")
    )
    for name in ("quant_conv", "post_quant_conv"):
        if config.get(f"use_{name}", False):
            modules.append((name, name))
    return Conversion({f"{old}.{p}": f"{new}.{p}" for old, new in modules for p in ("weight", "bias")})


def unwrap_cogvideox_checkpoint(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Select the tensor dictionary from the nested wrappers used by SAT and converted component checkpoints."""
    while True:
        wrappers = [key for key in ("model", "module", "state_dict") if isinstance(checkpoint.get(key), Mapping)]
        if not wrappers:
            return checkpoint
        if len(wrappers) != 1:
            raise ValueError(f"Ambiguous CogVideoX checkpoint wrappers: {wrappers}.")
        checkpoint = checkpoint[wrappers[0]]


def convert_cogvideox_transformer_checkpoint_to_diffusers(checkpoint, config):
    """Import a SAT transformer component, selecting its prefix and reporting omitted auxiliary keys."""
    checkpoint = unwrap_cogvideox_checkpoint(checkpoint)
    conversion = cogvideox_transformer_conversion(config)
    if any(key.startswith(COGVIDEOX_TRANSFORMER_PREFIX) for key in checkpoint):
        if conversion.original_keys.intersection(checkpoint):
            raise ValueError("CogVideoX checkpoint mixes prefixed and unprefixed transformer weights.")
        checkpoint = {
            key.removeprefix(COGVIDEOX_TRANSFORMER_PREFIX): tensor
            for key, tensor in checkpoint.items()
            if key.startswith(COGVIDEOX_TRANSFORMER_PREFIX)
        }
    else:
        checkpoint = dict(checkpoint)

    omitted = sorted(cogvideox_transformer_auxiliary_keys(config).intersection(checkpoint))
    for key in omitted:
        checkpoint.pop(key)
    if omitted:
        logger.info(f"Omitting CogVideoX auxiliary embeddings computed or unused by Diffusers: {omitted}")
    return conversion.to_diffusers(checkpoint)


def cogvideox_transformer_auxiliary_keys(config):
    """Known SAT embeddings computed or unused by the Diffusers component."""
    auxiliary_keys = {
        "transformer.embed_tokens.weight",
        "transformer.word_embeddings.weight",
        "transformer.position_embeddings.weight",
        "mixins.pos_embed.freqs_sin",
        "mixins.pos_embed.freqs_cos",
    }
    if not config.get("use_learned_positional_embeddings", False):
        auxiliary_keys.add("mixins.pos_embed.pos_embedding")
    return auxiliary_keys


def convert_cogvideox_vae_checkpoint_to_diffusers(checkpoint, config):
    """Import SAT VAE weights, excluding the training loss module without changing the input checkpoint."""
    checkpoint = unwrap_cogvideox_checkpoint(checkpoint)
    omitted = sorted(key for key in checkpoint if key.startswith("loss."))
    if omitted:
        logger.info(f"Omitting CogVideoX VAE training loss state: {omitted}")
    checkpoint = {key: tensor for key, tensor in checkpoint.items() if not key.startswith("loss.")}
    return cogvideox_vae_conversion(config).to_diffusers(checkpoint)
