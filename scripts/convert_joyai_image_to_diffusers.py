#!/usr/bin/env python3

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

from diffusers import JoyAIImagePipeline
from diffusers.configuration_utils import FrozenDict
from diffusers.models.autoencoders.autoencoder_kl_joyai_image import JoyAIImageVAE
from diffusers.models.transformers.transformer_joyai_image import JoyAIImageTransformer3DModel
from diffusers.schedulers.scheduling_joyai_flow_match_discrete import JoyAIFlowMatchDiscreteScheduler


DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class JoyAIImageSourceConfig:
    source_root: Path
    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    text_encoder_precision: str = "bf16"
    text_token_max_length: int = 2048
    enable_multi_task_training: bool = False
    dit_arch_config: dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_size": 4096,
            "in_channels": 16,
            "heads_num": 32,
            "mm_double_blocks_depth": 40,
            "out_channels": 16,
            "patch_size": [1, 2, 2],
            "rope_dim_list": [16, 56, 56],
            "text_states_dim": 4096,
            "rope_type": "rope",
            "dit_modulation_type": "wanx",
            "theta": 10000,
            "attn_backend": "flash_attn",
        }
    )
    scheduler_arch_config: dict[str, Any] = field(
        default_factory=lambda: {
            "num_train_timesteps": 1000,
            "shift": 4.0,
        }
    )

    @property
    def text_encoder_arch_config(self) -> dict[str, Any]:
        return {"params": {"text_encoder_ckpt": str(self.source_root / "JoyAI-Image-Und")}}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a raw JoyAI checkpoint directory to standard diffusers format."
    )
    parser.add_argument(
        "--source_path", type=str, required=True, help="Path to the original JoyAI checkpoint directory"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output path for the converted diffusers checkpoint"
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=sorted(DTYPE_MAP), help="Component dtype to load and save"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device used while loading the raw JoyAI checkpoint")
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        default=True,
        help="Save diffusers weights with safetensors when supported (default: True)",
    )
    return parser.parse_args()


def dtype_to_precision(torch_dtype: Optional[torch.dtype]) -> Optional[str]:
    if torch_dtype is None:
        return None
    for name, value in PRECISION_TO_TYPE.items():
        if value == torch_dtype and name in {"fp32", "fp16", "bf16"}:
            return name
    raise ValueError(f"Unsupported torch dtype for JoyAI conversion: {torch_dtype}")


def resolve_manifest_path(source_root: Path, manifest_value: Optional[str]) -> Optional[Path]:
    if manifest_value is None:
        return None
    path = Path(manifest_value)
    if path.parts and path.parts[0] == source_root.name:
        path = Path(*path.parts[1:])
    return source_root / path


def is_joyai_source_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "infer_config.py").is_file()
        and (path / "manifest.json").is_file()
        and (path / "transformer").is_dir()
        and (path / "vae").is_dir()
    )


def load_transformer_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    return state


def load_joyai_components(
    source_root: Union[str, Path],
    torch_dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> dict[str, Any]:
    source_root = Path(source_root)
    if not is_joyai_source_dir(source_root):
        raise ValueError(f"Not a valid JoyAI source checkpoint directory: {source_root}")

    precision = dtype_to_precision(torch_dtype)
    cfg = JoyAIImageSourceConfig(source_root=source_root)

    manifest = json.loads((source_root / "manifest.json").read_text())
    transformer_ckpt = resolve_manifest_path(source_root, manifest.get("transformer_ckpt"))
    vae_ckpt = source_root / "vae" / "Wan2.1_VAE.pth"
    text_encoder_ckpt = source_root / "JoyAI-Image-Und"

    if precision is not None:
        cfg.dit_precision = precision
        cfg.vae_precision = precision
        cfg.text_encoder_precision = precision

    load_device = torch.device(device) if device is not None else torch.device("cpu")
    transformer = JoyAIImageTransformer3DModel(
        dtype=PRECISION_TO_TYPE[cfg.dit_precision],
        device=load_device,
        **cfg.dit_arch_config,
    )
    state_dict = load_transformer_state_dict(transformer_ckpt)
    if "img_in.weight" in state_dict and transformer.img_in.weight.shape != state_dict["img_in.weight"].shape:
        value = state_dict["img_in.weight"]
        padded = value.new_zeros(transformer.img_in.weight.shape)
        padded[:, : value.shape[1], :, :, :] = value
        state_dict["img_in.weight"] = padded
    transformer.load_state_dict(state_dict, strict=True)
    transformer = transformer.to(dtype=PRECISION_TO_TYPE[cfg.dit_precision]).eval()

    vae = JoyAIImageVAE(
        pretrained=str(vae_ckpt),
        torch_dtype=PRECISION_TO_TYPE[cfg.vae_precision],
        device=load_device,
    )
    vae = vae.to(device=load_device, dtype=PRECISION_TO_TYPE[cfg.vae_precision]).eval()
    text_encoder = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            str(text_encoder_ckpt),
            dtype=PRECISION_TO_TYPE[cfg.text_encoder_precision],
            local_files_only=True,
            trust_remote_code=True,
        )
        .to(load_device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(text_encoder_ckpt),
        local_files_only=True,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        str(text_encoder_ckpt),
        local_files_only=True,
        trust_remote_code=True,
    )
    scheduler = JoyAIFlowMatchDiscreteScheduler(**cfg.scheduler_arch_config)

    return {
        "args": cfg,
        "processor": processor,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "transformer": transformer,
        "scheduler": scheduler,
        "vae": vae,
    }


def _sanitize_config_value(value: Any) -> Any:
    if isinstance(value, (torch.dtype, torch.device)):
        raise TypeError("Drop non-JSON torch config values")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            try:
                sanitized[key] = _sanitize_config_value(item)
                json.dumps(sanitized[key])
            except TypeError:
                continue
        return sanitized
    if isinstance(value, (list, tuple)):
        sanitized = []
        for item in value:
            try:
                converted = _sanitize_config_value(item)
                json.dumps(converted)
                sanitized.append(converted)
            except TypeError:
                continue
        return sanitized
    return value


def _sanitize_component_config(component: Any) -> None:
    config = getattr(component, "config", None)
    if config is None:
        return

    sanitized_config = {}
    for key, value in dict(config).items():
        try:
            sanitized_value = _sanitize_config_value(value)
            json.dumps(sanitized_value)
            sanitized_config[key] = sanitized_value
        except TypeError:
            continue

    component._internal_dict = FrozenDict(sanitized_config)


def _sanitize_pipeline_for_export(pipeline: JoyAIImagePipeline) -> None:
    for component_name in ["vae", "transformer", "scheduler"]:
        _sanitize_component_config(getattr(pipeline, component_name, None))


def main():
    args = parse_args()
    source_path = Path(args.source_path)
    output_path = Path(args.output_path)

    if not source_path.exists():
        raise ValueError(f"Source path does not exist: {source_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    components = load_joyai_components(
        source_root=source_path,
        torch_dtype=DTYPE_MAP[args.dtype],
        device=args.device,
    )
    pipeline = JoyAIImagePipeline(
        vae=components["vae"],
        text_encoder=components["text_encoder"],
        tokenizer=components["tokenizer"],
        transformer=components["transformer"],
        scheduler=components["scheduler"],
        processor=components["processor"],
        args=components["args"],
    )
    _sanitize_pipeline_for_export(pipeline)
    pipeline.save_pretrained(output_path, safe_serialization=args.safe_serialization)

    print(f"Converted JoyAI checkpoint saved to: {output_path}")
    print(f"Load with: JoyAIImagePipeline.from_pretrained({str(output_path)!r})")


if __name__ == "__main__":
    main()
