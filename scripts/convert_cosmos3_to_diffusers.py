#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Convert a Cosmos3 DCP checkpoint to diffusers format.

Example:
CUDA_VISIBLE_DEVICES=0 python scripts/convert_cosmos3_to_diffusers.py \
    --checkpoint-path Cosmos3-Nano \
    --output converted/cosmos3-nano-pipeline \
    --save-pipeline
"""

import argparse
import contextlib
import json
import pathlib
import re

import torch
from cosmos3.common.init import init_script


init_script()

from accelerate import init_empty_weights  # noqa: E402
from cosmos3.args import _CHECKPOINTS  # noqa: E402
from cosmos3.model import Cosmos3OmniModel  # noqa: E402
from projects.cosmos3.vfm.models.omni_mot_model import OmniMoTModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from diffusers import AutoencoderKLWan, UniPCMultistepScheduler  # noqa: E402
from diffusers.models.autoencoders.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer  # noqa: E402
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer  # noqa: E402
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniDiffusersPipeline  # noqa: E402


DEFAULT_SOUND_TOKENIZER_CONFIG = {
    "sampling_rate": 48000,
    "vocoder_input_dim": 64,
    "dec_dim": 320,
    "dec_c_mults": [1, 2, 4, 8, 16],
    "dec_strides": [2, 4, 5, 6, 8],
    "dec_out_channels": 2,
}


def _get_config_value(*configs, name, default=None):
    for config in configs:
        if config is None:
            continue
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
        if isinstance(config, dict) and config.get(name) is not None:
            return config[name]
    return default


def _load_sound_tokenizer_state_dict(checkpoint_path: pathlib.Path) -> dict[str, torch.Tensor]:
    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Loading AVAE .safetensors checkpoints requires safetensors.") from exc
        checkpoint = load_file(str(checkpoint_path), device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError(f"AVAE checkpoint must be a dict, got {type(checkpoint)!r}.")

    for key in ("generator", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break

    state_dict = {
        key: value.detach().cpu().contiguous() for key, value in checkpoint.items() if isinstance(value, torch.Tensor)
    }
    if not state_dict:
        raise RuntimeError(f"No tensor state dict found in AVAE checkpoint keys: {list(checkpoint.keys())[:16]}")
    return state_dict


def _load_sound_tokenizer_config(config_path: pathlib.Path | None, fallback_config_path: pathlib.Path) -> dict:
    selected_config_path = config_path
    if selected_config_path is None and fallback_config_path.exists():
        selected_config_path = fallback_config_path
    if selected_config_path is None:
        return dict(DEFAULT_SOUND_TOKENIZER_CONFIG)
    with open(selected_config_path, encoding="utf-8") as f:
        return json.load(f)


_SOUND_TOKENIZER_PER_KEY_PREFIXES = ("module.", "generator.", "model.", "state_dict.")
_SOUND_TOKENIZER_RES_UNIT_INNER_NAMES = {0: "snake1", 1: "conv1", 2: "snake2", 3: "conv2"}


def _sound_tokenizer_strip_per_key_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(state_dict)
    changed = True
    while changed:
        changed = False
        for prefix in _SOUND_TOKENIZER_PER_KEY_PREFIXES:
            if any(key.startswith(prefix) for key in out):
                out = {(key[len(prefix) :] if key.startswith(prefix) else key): value for key, value in out.items()}
                changed = True
                break
        if any(key.startswith(("decoder.", "encoder.", "bottleneck.")) for key in out):
            break
    return out


def _sound_tokenizer_filter_decoder(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state_dict.items() if key.startswith("decoder.")}


def _sound_tokenizer_infer_num_blocks(state_dict: dict[str, torch.Tensor]) -> int:
    block_indices: set[int] = set()
    for key in state_dict:
        match = re.match(r"decoder\.layers\.(\d+)\.layers\.\d+\.", key)
        if match:
            block_indices.add(int(match.group(1)))
    return len(block_indices)


def _sound_tokenizer_remap_flat_layout(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert legacy AVAE `decoder.layers.*` keys to OobleckDecoder attribute keys."""
    if not any(re.match(r"decoder\.layers\.\d+\.", key) for key in state_dict):
        return state_dict

    num_blocks = _sound_tokenizer_infer_num_blocks(state_dict)
    if num_blocks == 0:
        raise RuntimeError("Detected flat `decoder.layers.*` layout but no decoder blocks were found; cannot remap.")
    snake1_idx = num_blocks + 1
    conv2_idx = num_blocks + 2

    def _remap(key: str) -> str:
        match = re.fullmatch(r"decoder\.layers\.(\d+)\.layers\.(\d+)\.layers\.(\d+)\.(.+)", key)
        if match:
            block_n, res_n, inner_n, rest = (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                match.group(4),
            )
            if res_n not in (2, 3, 4):
                raise RuntimeError(f"Unexpected residual position {res_n} in {key!r}.")
            inner_name = _SOUND_TOKENIZER_RES_UNIT_INNER_NAMES.get(inner_n)
            if inner_name is None:
                raise RuntimeError(f"Unexpected residual inner index {inner_n} in {key!r}.")
            return f"decoder.block.{block_n - 1}.res_unit{res_n - 1}.{inner_name}.{rest}"

        match = re.fullmatch(r"decoder\.layers\.(\d+)\.layers\.(\d+)\.(.+)", key)
        if match:
            block_n, sub_n, rest = int(match.group(1)), int(match.group(2)), match.group(3)
            block_idx = block_n - 1
            if sub_n == 0:
                return f"decoder.block.{block_idx}.snake1.{rest}"
            if sub_n == 1:
                return f"decoder.block.{block_idx}.conv_t1.{rest}"
            raise RuntimeError(f"Unexpected decoder block sub-index {sub_n} in {key!r}.")

        match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", key)
        if match:
            layer_n, rest = int(match.group(1)), match.group(2)
            if layer_n == 0:
                return f"decoder.conv1.{rest}"
            if layer_n == snake1_idx:
                return f"decoder.snake1.{rest}"
            if layer_n == conv2_idx:
                return f"decoder.conv2.{rest}"
            raise RuntimeError(
                f"Unexpected decoder leaf layer index {layer_n} (expected 0, {snake1_idx}, or {conv2_idx}) in {key!r}."
            )

        return key

    return {_remap(key): value for key, value in state_dict.items()}


def _sound_tokenizer_reshape_snake_params(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if (key.endswith(".alpha") or key.endswith(".beta")) and value.ndim == 1:
            value = value.unsqueeze(0).unsqueeze(-1).contiguous()
        out[key] = value
    return out


def _sound_tokenizer_reapply_weight_norm(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reconstruct weight-norm parameters if the source checkpoint has folded conv weights."""
    out = dict(state_dict)
    candidate_keys = [
        key
        for key in state_dict
        if key.endswith(".weight") and any(f".{layer}." in key for layer in ("conv1", "conv2", "conv_t1"))
    ]
    for key in candidate_keys:
        stem = key[: -len(".weight")]
        weight_g_key = f"{stem}.weight_g"
        weight_v_key = f"{stem}.weight_v"
        if weight_g_key in state_dict or weight_v_key in state_dict:
            continue
        weight = state_dict[key]
        norm_dims = tuple(range(1, weight.ndim))
        out.pop(key)
        out[weight_g_key] = weight.norm(p=2, dim=norm_dims, keepdim=True).contiguous()
        out[weight_v_key] = weight.contiguous()
    return out


def _remap_avae_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a legacy AVAE state dict into the Cosmos3AVAEAudioTokenizer state dict."""
    state_dict = _sound_tokenizer_strip_per_key_prefixes(state_dict)
    state_dict = _sound_tokenizer_filter_decoder(state_dict)
    if not state_dict:
        raise RuntimeError("Sound tokenizer state dict has no `decoder.*` keys after prefix stripping.")
    state_dict = _sound_tokenizer_remap_flat_layout(state_dict)
    state_dict = _sound_tokenizer_reshape_snake_params(state_dict)
    state_dict = _sound_tokenizer_reapply_weight_norm(state_dict)
    if any(re.match(r"decoder\.layers\.\d+", key) for key in state_dict):
        raise RuntimeError("Flat `decoder.layers.*` keys remain after remap; conversion is incomplete.")
    return state_dict


def _build_sound_tokenizer(
    checkpoint_path: pathlib.Path,
    config_path: pathlib.Path | None,
) -> Cosmos3AVAEAudioTokenizer:
    config = _load_sound_tokenizer_config(config_path, fallback_config_path=pathlib.Path())
    print(f"Loading AVAE sound tokenizer weights from {checkpoint_path} …")
    raw_state_dict = _load_sound_tokenizer_state_dict(checkpoint_path)
    state_dict = _remap_avae_state_dict(raw_state_dict)
    print(f"  Remapped {len(raw_state_dict)} → {len(state_dict)} decoder keys.")

    sound_tokenizer = Cosmos3AVAEAudioTokenizer(
        sampling_rate=config.get("sampling_rate", DEFAULT_SOUND_TOKENIZER_CONFIG["sampling_rate"]),
        vocoder_input_dim=config.get("vocoder_input_dim", DEFAULT_SOUND_TOKENIZER_CONFIG["vocoder_input_dim"]),
        dec_dim=config.get("dec_dim", DEFAULT_SOUND_TOKENIZER_CONFIG["dec_dim"]),
        dec_c_mults=tuple(config.get("dec_c_mults", DEFAULT_SOUND_TOKENIZER_CONFIG["dec_c_mults"])),
        dec_strides=tuple(config.get("dec_strides", DEFAULT_SOUND_TOKENIZER_CONFIG["dec_strides"])),
        dec_out_channels=config.get("dec_out_channels", DEFAULT_SOUND_TOKENIZER_CONFIG["dec_out_channels"]),
    )
    load_result = sound_tokenizer.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Cosmos3 AVAE sound tokenizer load did not match strictly: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}."
        )
    return sound_tokenizer


@contextlib.contextmanager
def _skip_source_sound_tokenizer_load():
    original_set_up_tokenizers = OmniMoTModel.set_up_tokenizers

    def set_up_tokenizers_without_sound(self):
        if not getattr(self.config, "sound_gen", False):
            return original_set_up_tokenizers(self)

        sound_gen = self.config.sound_gen
        self.config.sound_gen = False
        try:
            return original_set_up_tokenizers(self)
        finally:
            self.config.sound_gen = sound_gen

    OmniMoTModel.set_up_tokenizers = set_up_tokenizers_without_sound
    try:
        yield
    finally:
        OmniMoTModel.set_up_tokenizers = original_set_up_tokenizers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        default="Cosmos3-Nano",
        help="Named checkpoint (e.g. 'Cosmos3-Nano') or path to DCP checkpoint dir.",
    )
    parser.add_argument("--output", required=True, help="Directory to save the converted diffusers model.")
    parser.add_argument(
        "--save-pipeline",
        action="store_true",
        help="Save the full pipeline (transformer + VAE + tokenizer + scheduler).",
    )
    parser.add_argument(
        "--dtype", default="bf16", choices=["fp32", "fp16", "bf16"], help="Dtype to save the transformer in."
    )
    parser.add_argument(
        "--sound-tokenizer-path", help="Optional AVAE sound tokenizer checkpoint to save under sound_tokenizer/."
    )
    parser.add_argument(
        "--sound-tokenizer-config-path", help="Optional AVAE config JSON to save under sound_tokenizer/config.json."
    )
    parser.add_argument(
        "--include-sound-tokenizer",
        action="store_true",
        help="Require saving sound_tokenizer/ even if the source transformer is video-only.",
    )
    args = parser.parse_args()

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    sound_tokenizer_path = (
        pathlib.Path(args.sound_tokenizer_path).expanduser().absolute() if args.sound_tokenizer_path else None
    )
    sound_tokenizer_config_path = (
        pathlib.Path(args.sound_tokenizer_config_path).expanduser().absolute()
        if args.sound_tokenizer_config_path
        else None
    )
    if args.include_sound_tokenizer and sound_tokenizer_path is None:
        raise ValueError("Sound tokenizer output was requested, but --sound-tokenizer-path was not provided.")
    if sound_tokenizer_path is not None and not sound_tokenizer_path.exists():
        raise FileNotFoundError(f"Sound tokenizer checkpoint not found: {sound_tokenizer_path}")
    if sound_tokenizer_config_path is not None and not sound_tokenizer_config_path.exists():
        raise FileNotFoundError(f"Sound tokenizer config not found: {sound_tokenizer_config_path}")

    checkpoint_name = args.checkpoint_path
    if checkpoint_name in _CHECKPOINTS:
        checkpoint_path = pathlib.Path(_CHECKPOINTS[checkpoint_name].download())
    else:
        checkpoint_path = pathlib.Path(checkpoint_name).expanduser().absolute()
    print(f"Resolved checkpoint path: {checkpoint_path}")

    print("Instantiating model and loading weights from DCP checkpoint …")
    print("Skipping source AVAE tokenizer instantiation during converter-only model load …")
    with _skip_source_sound_tokenizer_load():
        _tmp = Cosmos3OmniModel.from_pretrained_dcp(checkpoint_path).model

    # Extract network components and architecture config from DCP model
    language_model = _tmp.net.language_model
    vae2llm = _tmp.net.vae2llm
    llm2vae = _tmp.net.llm2vae
    time_embedder = _tmp.net.time_embedder
    lm_cfg = _tmp.net.language_model.config
    net_cfg = _tmp.net.config
    model_cfg = _tmp.config
    patch_latent_dim = _tmp.net.patch_latent_dim
    hidden_size = _tmp.net.hidden_size
    num_attention_heads = _tmp.net.num_heads
    num_key_value_heads = _tmp.net.num_kv_heads
    head_dim = _tmp.net.head_dim
    num_hidden_layers = _tmp.net.num_hidden_layers
    latent_patch_size = _tmp.net.latent_patch_size
    latent_channel = _tmp.net.latent_channel
    timestep_scale = _tmp.net.timestep_scale
    use_moe = _tmp.net.use_moe
    joint_attn_implementation = net_cfg.joint_attn_implementation
    base_fps = int(net_cfg.base_fps)
    enable_fps_modulation = net_cfg.enable_fps_modulation
    max_action_dim = _tmp.config.max_action_dim
    position_embedding_type = net_cfg.position_embedding_type
    unified_3d_mrope_reset_spatial_ids = _tmp.config.diffusion_expert_config.unified_3d_mrope_reset_spatial_ids
    unified_3d_mrope_temporal_modality_margin = (
        _tmp.config.diffusion_expert_config.unified_3d_mrope_temporal_modality_margin
    )
    video_temporal_causal = net_cfg.video_temporal_causal
    sound2llm = getattr(_tmp.net, "sound2llm", None)
    llm2sound = getattr(_tmp.net, "llm2sound", None)
    sound_modality_embed = getattr(_tmp.net, "sound_modality_embed", None)
    has_sound_projection_weights = any(module is not None for module in (sound2llm, llm2sound, sound_modality_embed))
    sound_gen = bool(
        _get_config_value(net_cfg, model_cfg, name="sound_gen", default=False) or has_sound_projection_weights
    )
    sound_dim = _get_config_value(net_cfg, model_cfg, name="sound_dim", default=None)
    if sound_dim is None and sound2llm is not None:
        sound_dim = sound2llm.in_features
    sound_latent_fps = _get_config_value(net_cfg, model_cfg, name="sound_latent_fps", default=25.0)
    if sound_gen:
        missing_sound_modules = [
            name
            for name, module in (
                ("sound2llm", sound2llm),
                ("llm2sound", llm2sound),
                ("sound_modality_embed", sound_modality_embed),
            )
            if module is None
        ]
        if missing_sound_modules:
            raise RuntimeError(
                "Source checkpoint is configured for sound generation but is missing "
                f"sound projection weights: {missing_sound_modules}."
            )
        if sound_dim is None:
            raise RuntimeError("Source checkpoint is configured for sound generation but sound_dim is missing.")
    del _tmp
    torch.cuda.empty_cache()

    # Init diffusers Cosmos3OmniTransformer with full architecture config from DCP
    with init_empty_weights():
        transformer = Cosmos3OmniTransformer(
            attention_bias=lm_cfg.attention_bias,
            attention_dropout=lm_cfg.attention_dropout,
            base_fps=base_fps,
            enable_fps_modulation=enable_fps_modulation,
            head_dim=head_dim,
            hidden_size=hidden_size,
            intermediate_size=lm_cfg.intermediate_size,
            joint_attn_implementation=joint_attn_implementation,
            latent_channel=latent_channel,
            latent_patch_size=latent_patch_size,
            max_action_dim=max_action_dim,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            patch_latent_dim=patch_latent_dim,
            position_embedding_type=position_embedding_type,
            rms_norm_eps=lm_cfg.rms_norm_eps,
            rope_scaling=lm_cfg.rope_scaling,
            rope_theta=lm_cfg.rope_theta,
            sound_dim=sound_dim,
            sound_gen=sound_gen,
            sound_latent_fps=sound_latent_fps,
            timestep_scale=timestep_scale,
            unified_3d_mrope_reset_spatial_ids=unified_3d_mrope_reset_spatial_ids,
            unified_3d_mrope_temporal_modality_margin=unified_3d_mrope_temporal_modality_margin,
            use_moe=use_moe,
            video_temporal_causal=video_temporal_causal,
            vocab_size=lm_cfg.vocab_size,
        )
    state_dict = language_model.state_dict()
    for k, v in vae2llm.state_dict().items():
        state_dict[f"vae2llm.{k}"] = v
    for k, v in llm2vae.state_dict().items():
        state_dict[f"llm2vae.{k}"] = v
    _TIME_EMBEDDER_REMAP = {
        "mlp.0.weight": "linear_1.weight",
        "mlp.0.bias": "linear_1.bias",
        "mlp.2.weight": "linear_2.weight",
        "mlp.2.bias": "linear_2.bias",
    }
    for k, v in time_embedder.state_dict().items():
        state_dict[f"time_embedder.{_TIME_EMBEDDER_REMAP[k]}"] = v
    if sound_gen:
        for k, v in sound2llm.state_dict().items():
            state_dict[f"sound2llm.{k}"] = v
        for k, v in llm2sound.state_dict().items():
            state_dict[f"llm2sound.{k}"] = v
        state_dict["sound_modality_embed"] = sound_modality_embed
    transformer.load_state_dict(state_dict, strict=True, assign=True)
    del (
        language_model,
        vae2llm,
        llm2vae,
        time_embedder,
        sound2llm,
        llm2sound,
        sound_modality_embed,
        state_dict,
    )
    torch.cuda.empty_cache()

    transformer = transformer.to(dtype=dtype)

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_sound_tokenizer = (
        args.include_sound_tokenizer or sound_tokenizer_path is not None or (sound_gen and args.save_pipeline)
    )
    if include_sound_tokenizer and sound_tokenizer_path is None:
        raise ValueError(
            "The source checkpoint is configured for sound generation, so --sound-tokenizer-path "
            "is required when saving a full pipeline."
        )

    if args.save_pipeline:
        text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

        diffusers_vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", torch_dtype=torch.bfloat16
        )
        sound_tokenizer = None
        if include_sound_tokenizer:
            assert sound_tokenizer_path is not None
            sound_tokenizer = _build_sound_tokenizer(sound_tokenizer_path, sound_tokenizer_config_path)

        # Karras schedule approximating FlowUniPCMultistepScheduler with shift=5, 35 steps.
        # Measured from that schedule: first flow-sigma=0.9998, last flow-sigma=0.1281.
        # EDM sigma = flow_sigma / (1 - flow_sigma), so:
        #   sigma_max = 0.9998 / 0.0002 = 4999  (but capped at 200 to avoid duplicate
        #               integer timesteps from Karras clustering near the top)
        #   sigma_min = 0.1281 / (1 - 0.1281)  = 0.1281 / 0.8719 ≈ 0.147
        scheduler = UniPCMultistepScheduler(
            use_karras_sigmas=True,
            use_flow_sigmas=True,
            prediction_type="flow_prediction",
            sigma_max=200.0,
            sigma_min=0.147,
        )

        pipeline = Cosmos3OmniDiffusersPipeline(
            transformer=transformer,
            text_tokenizer=text_tokenizer,
            vae=diffusers_vae,
            scheduler=scheduler,
            sound_tokenizer=sound_tokenizer,
        )
        print(f"Saving full pipeline to {output_dir} …")
        pipeline.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size="5GB")
    else:
        print(f"Saving transformer to {output_dir} …")
        transformer.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size="5GB")
        if include_sound_tokenizer:
            print("Skipping sound_tokenizer/ save because --save-pipeline was not set.")

    print("Done.")


if __name__ == "__main__":
    main()
