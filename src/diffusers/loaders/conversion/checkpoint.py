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

"""Adapt published single-file component containers to the strict tensor conversion API."""

from collections.abc import Iterator, Mapping
from typing import Any

import torch

from ...utils import logging
from .cogvideox import (
    convert_cogvideox_transformer_checkpoint_to_diffusers,
    convert_cogvideox_vae_checkpoint_to_diffusers,
)
from .core import Conversion
from .registry import get_conversion


logger = logging.get_logger(__name__)


class ComponentState(Mapping[str, Any]):
    """Select a component without reading its tensors until the conversion needs them."""

    def __init__(self, checkpoint: Mapping[str, Any], prefix: str = "") -> None:
        self.checkpoint = checkpoint
        self.keys_to_source = {key[len(prefix) :]: key for key in checkpoint if key.startswith(prefix)}
        self.generated = {}

    def __getitem__(self, key: str) -> Any:
        if key in self.generated:
            return self.generated[key]
        return self.checkpoint[self.keys_to_source[key]]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys_to_source.keys() | self.generated.keys())

    def __len__(self) -> int:
        return len(self.keys_to_source.keys() | self.generated.keys())


def convert_component_checkpoint(
    checkpoint: Mapping[str, Any],
    config: dict[str, Any],
    model_class: str,
    *,
    extract_ema: bool = False,
    return_config: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Import a single-file component with its resolved Diffusers config, without mutating the checkpoint.

    Prefix selection and the known auxiliary keys formerly handled in `single_file_utils` live here. The conversion
    definition itself remains a strict, reversible mapping of component tensors. Only explicitly known auxiliary state
    is excluded; unknown keys inside the selected component raise instead of silently disappearing. With
    `return_config=True`, also return the resolved config so callers can persist inferred original formats.
    """
    config = dict(config)
    if model_class == "CogVideoXTransformer3DModel":
        converted = convert_cogvideox_transformer_checkpoint_to_diffusers(checkpoint, config=config)
    elif model_class == "AutoencoderKLCogVideoX":
        converted = convert_cogvideox_vae_checkpoint_to_diffusers(checkpoint, config=config)
    else:
        state, conversion, config = prepare_component_checkpoint(
            checkpoint, config, model_class, extract_ema=extract_ema
        )
        converted = conversion.to_diffusers(state)
    return (converted, config) if return_config else converted


def prepare_component_checkpoint(
    checkpoint: Mapping[str, Any],
    config: dict[str, Any],
    model_class: str,
    *,
    extract_ema: bool = False,
) -> tuple[Mapping[str, Any], Conversion, dict[str, Any]]:
    """Return a lazy component view, its conversion, and the resolved config without mutating caller inputs."""

    config = dict(config)
    conversion = get_conversion(model_class, config)
    if checkpoint.keys() == conversion.original_keys and not extract_ema:
        return checkpoint, conversion, config

    # These definitions span several already-qualified namespaces. Keep the prefixes and reject unknown entries
    # inside the selected component, while leaving sibling components in the bundle alone.
    namespaces = ()
    if model_class == "AutoencoderKLTemporalDecoder" and config.get("original_format", "svd") == "svd":
        namespaces = ("conditioner.embedders.3.encoder.", "first_stage_model.")
    elif model_class == "UNetFlatConditionModel":
        namespaces = ("model.diffusion_model.time_embed.", "model.diffusion_model.unet_text.")
    elif model_class == "UNet2DConditionModel" and config.get("original_format") == "versatile_image":
        namespaces = ("model.diffusion_model.time_embed.", "model.diffusion_model.unet_image.")
    elif model_class == "PaintByExampleImageEncoder":
        namespaces = ("cond_stage_model.", "proj_out.", "learnable_vector")

    prefixes = {
        "UNet2DConditionModel": ("model.diffusion_model.",),
        "ControlNetModel": ("control_model.",),
        "AutoencoderKL": ("first_stage_model.", "vae."),
        "AutoencoderKLHunyuanImage": ("vae.",),
        "AutoencoderKLLTXVideo": ("vae.",),
        "AutoencoderKLLTX2Video": ("vae.",),
        "AutoencoderKLLTX2Audio": ("audio_vae.",),
        "CosmosTransformer3DModel": ("net.", "model.diffusion_model."),
        "MiniMaxMusic3Transformer1DModel": ("diffusion_transformer.",),
    }.get(model_class, ("model.diffusion_model.",))
    prefix = "" if namespaces else next((p for p in prefixes if any(key.startswith(p) for key in checkpoint)), "")
    state = ComponentState(checkpoint, prefix)
    if namespaces:
        state.keys_to_source = {
            key: value for key, value in state.keys_to_source.items() if key.startswith(namespaces)
        }
    if extract_ema:
        if not prefix and not namespaces:
            raise ValueError("EMA selection requires a bundled LDM checkpoint with model.diffusion_model keys.")
        ema_keys = {key: "model_ema." + "".join((prefix + key).split(".")[1:]) for key in state}
        if any(key not in checkpoint for key in ema_keys.values()):
            raise ValueError("The checkpoint does not contain a complete EMA copy of the UNet.")
        state.keys_to_source = ema_keys

    if model_class == "CosmosTransformer3DModel" and "original_format" not in config:
        config["original_format"] = "cosmos1" if any(key.startswith("blocks.block") for key in state) else "cosmos2"
    if model_class == "LTX2VideoDiffusionDecoderModel":
        if "original_format" not in config and any(
            key.endswith((".gate_msa", ".gate_mlp", ".gate_ctx")) for key in state
        ):
            config["original_format"] = "ltx2_diffusion_decoder_gated"
    conversion = get_conversion(model_class, config)
    if state.keys() == conversion.diffusers_keys:
        return state, Conversion(mapping={key: key for key in state}), config
    if model_class == "AutoencoderRAE" and "decoder.trainable_cls_token" not in state:
        anchor = state["decoder.decoder_embed.weight"]
        state.generated["decoder.trainable_cls_token"] = anchor.new_zeros((1, 1, anchor.shape[0]))
    if (
        model_class == "SkyReelsV2Transformer3DModel"
        and "img_emb.emb_pos" in conversion.original_keys
        and "img_emb.emb_pos" not in state
    ):
        state.generated["img_emb.emb_pos"] = state["img_emb.proj.0.weight"].new_zeros(
            1, config["pos_embed_seq_len"], config["image_dim"]
        )
    if model_class in ("ClapModel", "ClapTextModelWithProjection"):
        text_config = config.get("text_config", config)
        positions = text_config.get("max_position_embeddings", 514)
        for name in ("position_ids", "token_type_ids"):
            key = "text_branch.embeddings." + name
            if key in conversion.original_keys and key not in state:
                state.generated[key] = (
                    torch.arange(positions, dtype=torch.long).unsqueeze(0)
                    if name == "position_ids"
                    else torch.zeros((1, positions), dtype=torch.long)
                )
    if model_class == "LTX2VideoDiffusionDecoderModel":
        # Distilled checkpoints can gate only a subset of blocks. An omitted gate is the identity.
        for rule in conversion.rules:
            gate = rule.original[0]
            if gate.endswith((".gate_msa", ".gate_mlp", ".gate_ctx")) and gate not in state:
                weight = state[rule.original[1]]
                state.generated[gate] = weight.new_ones(weight.shape[0])
    auxiliary = set()
    auxiliary_prefixes = ()
    if model_class in ("AutoencoderKL", "VQModel"):
        auxiliary_prefixes = ("loss.", "vocoder.")
    elif model_class in ("ClapModel", "ClapTextModelWithProjection"):
        auxiliary_prefixes = ("text_transform.", "audio_transform.")
        auxiliary = {
            key
            for key in state
            if any(part in key.split(".") for part in ("stft", "logmel_extractor", "tscam_conv", "head", "attn_mask"))
        }
    elif model_class == "AsymmetricAutoencoderKL":
        auxiliary_prefixes = ("loss.", "decoder.up_layers.")
    elif model_class == "AutoencoderRAE":
        auxiliary = {"decoder.decoder_pos_embed"}
    elif model_class == "MiniMaxH3Transformer3DModel":
        auxiliary = {"rope.inv_freq"}
    elif model_class == "UNet1DModel":
        auxiliary = {key for key in state if key.endswith(".kernel")}
    elif model_class == "SanaControlNetModel":
        auxiliary_prefixes = ("blocks.", "final_layer.")
        auxiliary = {"pos_embed", "y_embedder.y_embedding", "logvar_linear.weight", "logvar_linear.bias"}
    elif model_class in (
        "SanaTransformer2DModel",
        "SanaVideoTransformer3DModel",
        "PixArtTransformer2DModel",
        "DiTTransformer2DModel",
        "Transformer2DModel",
    ):
        auxiliary = {"pos_embed", "y_embedder.y_embedding"}
        if model_class in ("SanaTransformer2DModel", "SanaVideoTransformer3DModel"):
            auxiliary.update(("logvar_linear.weight", "logvar_linear.bias"))
    elif model_class in ("AutoencoderKLLTXVideo", "AutoencoderKLLTX2Video"):
        auxiliary = {"per_channel_statistics.channel", "per_channel_statistics.mean-of-stds"}
    elif model_class == "LTX2VideoTransformer3DModel":
        auxiliary_prefixes = ("video_embeddings_connector.", "audio_embeddings_connector.")
    elif model_class == "LTX2VideoDiffusionDecoderModel":
        auxiliary_prefixes = ("encoder.", "decoder.coarse_")
        auxiliary = {"per_channel_statistics.channel", "per_channel_statistics.mean-of-stds"}
        auxiliary.update(key for key in state if key.startswith("decoder.") and ".coarse_" in key)
    elif model_class == "CosmosTransformer3DModel":
        auxiliary = {
            "logvar.0.freqs",
            "logvar.0.phases",
            "logvar.1.weight",
            "pos_embedder.seq",
            "pos_embedder.dim_spatial_range",
            "pos_embedder.dim_temporal_range",
            "_extra_state",
            "accum_video_sample_counter",
            "accum_image_sample_counter",
            "accum_iteration",
            "accum_train_in_hours",
        }
    elif model_class == "ZImageControlNetModel" and config.get("add_control_noise_refiner") == "control_layers":
        auxiliary_prefixes = ("control_noise_refiner.",)
    removed = [
        key
        for key in state
        if key not in conversion.original_keys and (key in auxiliary or key.startswith(auxiliary_prefixes))
    ]
    if removed:
        logger.info("Excluding known auxiliary state from %s: %s", model_class, sorted(removed))
        for key in removed:
            state.keys_to_source.pop(key, None)
            state.generated.pop(key, None)
    return state, conversion, config
