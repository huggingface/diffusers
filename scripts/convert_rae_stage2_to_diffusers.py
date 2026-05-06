import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import HfApi, hf_hub_download

from diffusers import AutoencoderRAE, FlowMatchEulerDiscreteScheduler, RAEDiTPipeline
from diffusers.models.transformers.transformer_rae_dit import RAEDiT2DModel
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME


DEFAULT_NUM_TRAIN_TIMESTEPS = 1000
DEFAULT_SHIFT_BASE = 4096
DEFAULT_TRANSFORMER_SUBFOLDER = "transformer"
DEFAULT_SCHEDULER_SUBFOLDER = "scheduler"


class RepoAccessor:
    def __init__(self, repo_or_path: str, cache_dir: str | None = None):
        self.repo_or_path = repo_or_path
        self.cache_dir = cache_dir
        self.local_root: Path | None = None
        self.repo_id: str | None = None
        self.repo_files: set[str] | None = None

        root = Path(repo_or_path)
        if root.exists() and root.is_dir():
            self.local_root = root
        else:
            self.repo_id = repo_or_path
            self.repo_files = set(HfApi().list_repo_files(repo_or_path))

    def exists(self, relative_path: str) -> bool:
        relative_path = relative_path.replace("\\", "/")
        if self.local_root is not None:
            return (self.local_root / relative_path).is_file()
        return relative_path in self.repo_files

    def fetch(self, relative_path: str) -> Path:
        relative_path = relative_path.replace("\\", "/")
        if self.local_root is not None:
            path = self.local_root / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"File not found: {path}")
            return path

        downloaded = hf_hub_download(repo_id=self.repo_id, filename=relative_path, cache_dir=self.cache_dir)
        return Path(downloaded)


def read_yaml(accessor: RepoAccessor, relative_path: str) -> dict[str, Any]:
    with resolve_input_path(accessor, relative_path).open() as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML object at `{relative_path}` to decode to a dictionary.")

    return config


def _get_nested(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Missing `{path}` in checkpoint/config object.")
        value = value[part]
    return value


def _resolve_section(config: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        section = config.get(key)
        if isinstance(section, dict):
            return section
    raise KeyError(f"Could not find any of {keys} in config.")


def _normalize_pair(value: int | list[int] | tuple[int, ...], field_name: str) -> tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"`{field_name}` must have length 2 when provided as a sequence, but got {value}.")
        return int(value[0]), int(value[1])

    scalar = int(value)
    return scalar, scalar


def _normalize_patch_size(value: int | list[int] | tuple[int, ...]) -> int | tuple[int, int]:
    stage1_patch_size, stage2_patch_size = _normalize_pair(value, "patch_size")
    if stage1_patch_size == stage2_patch_size:
        return stage1_patch_size
    return stage1_patch_size, stage2_patch_size


def _maybe_strip_common_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    if len(state_dict) > 0 and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def unwrap_state_dict(
    maybe_wrapped: dict[str, Any],
    checkpoint_key: str | None = None,
    prefer_ema: bool = True,
) -> dict[str, Any]:
    if checkpoint_key:
        state_dict = _get_nested(maybe_wrapped, checkpoint_key)
    else:
        state_dict = maybe_wrapped
        if isinstance(state_dict, dict):
            candidate_keys = ["ema", "model", "state_dict"] if prefer_ema else ["model", "ema", "state_dict"]
            for key in candidate_keys:
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break

    if not isinstance(state_dict, dict):
        raise ValueError("Resolved checkpoint payload is not a dictionary state dict.")

    state_dict = dict(state_dict)
    for prefix in ("model.module.", "model.", "module."):
        state_dict = _maybe_strip_common_prefix(state_dict, prefix)
    return state_dict


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    suffix = checkpoint_path.suffix.lower()
    if suffix == ".safetensors":
        import safetensors.torch

        return safetensors.torch.load_file(checkpoint_path)

    return torch.load(checkpoint_path, map_location="cpu")


def build_transformer_config(stage2_params: dict[str, Any], misc: dict[str, Any]) -> dict[str, Any]:
    hidden_size = _normalize_pair(stage2_params["hidden_size"], "hidden_size")
    depth = _normalize_pair(stage2_params["depth"], "depth")
    num_heads = _normalize_pair(stage2_params["num_heads"], "num_heads")
    patch_size = _normalize_patch_size(stage2_params.get("patch_size", 1))

    input_size = int(stage2_params["input_size"])
    in_channels = int(stage2_params["in_channels"])

    latent_size = misc.get("latent_size")
    if latent_size is not None:
        if len(latent_size) != 3:
            raise ValueError(f"`misc.latent_size` should have length 3, but got {latent_size}.")
        latent_channels, latent_height, latent_width = [int(dim) for dim in latent_size]
        if latent_channels != in_channels:
            raise ValueError(
                f"`misc.latent_size[0]` ({latent_channels}) does not match `stage_2.params.in_channels` ({in_channels})."
            )
        if latent_height != input_size or latent_width != input_size:
            raise ValueError(
                f"`misc.latent_size[1:]` ({latent_height}, {latent_width}) does not match `stage_2.params.input_size` ({input_size})."
            )

    return {
        "sample_size": input_size,
        "patch_size": patch_size,
        "in_channels": in_channels,
        "hidden_size": hidden_size,
        "depth": depth,
        "num_heads": num_heads,
        "mlp_ratio": float(stage2_params.get("mlp_ratio", 4.0)),
        "class_dropout_prob": float(stage2_params.get("class_dropout_prob", 0.1)),
        "num_classes": int(stage2_params.get("num_classes", misc.get("num_classes", 1000))),
        "use_qknorm": bool(stage2_params.get("use_qknorm", False)),
        "use_swiglu": bool(stage2_params.get("use_swiglu", True)),
        "use_rope": bool(stage2_params.get("use_rope", True)),
        "use_rmsnorm": bool(stage2_params.get("use_rmsnorm", True)),
        "wo_shift": bool(stage2_params.get("wo_shift", False)),
        "use_pos_embed": bool(stage2_params.get("use_pos_embed", True)),
    }


def build_scheduler_config(config: dict[str, Any]) -> tuple[FlowMatchEulerDiscreteScheduler, dict[str, Any]]:
    transport = _resolve_section(config, "transport")
    misc = _resolve_section(config, "misc")

    transport_params = transport.get("params", {})
    path_type = str(transport_params.get("path_type", "Linear"))
    prediction = str(transport_params.get("prediction", "velocity"))
    if path_type.lower() != "linear" or prediction.lower() != "velocity":
        raise ValueError(
            "Only `transport.params.path_type=Linear` with `transport.params.prediction=velocity` is "
            "supported by this converter because it always saves a `FlowMatchEulerDiscreteScheduler`."
        )

    latent_size = misc.get("latent_size", None)
    if latent_size is None:
        raise KeyError("Config must define `misc.latent_size` for scheduler conversion.")

    shift_dim = int(misc.get("time_dist_shift_dim", math.prod(int(dim) for dim in latent_size)))
    shift_base = int(misc.get("time_dist_shift_base", DEFAULT_SHIFT_BASE))
    shift = math.sqrt(shift_dim / shift_base)

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=int(transport_params.get("num_train_timesteps", DEFAULT_NUM_TRAIN_TIMESTEPS)),
        shift=shift,
        stochastic_sampling=False,
    )
    metadata = {
        "num_train_timesteps": scheduler.config.num_train_timesteps,
        "shift": scheduler.config.shift,
        "path_type": path_type,
        "prediction": prediction,
        "time_dist_type": transport_params.get("time_dist_type", "uniform"),
    }
    return scheduler, metadata


def _swap_projection_halves(tensor: torch.Tensor) -> torch.Tensor:
    return torch.cat(tensor.chunk(2, dim=0)[::-1], dim=0)


def translate_transformer_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    translated = {}

    for key, value in state_dict.items():
        if ".mlp.w12." in key:
            new_key = key.replace(".mlp.w12.", ".mlp.net.0.proj.")
            if isinstance(value, torch.Tensor):
                value = _swap_projection_halves(value)
        elif ".mlp.w3." in key:
            new_key = key.replace(".mlp.w3.", ".mlp.net.2.")
        elif ".mlp.fc1." in key:
            new_key = key.replace(".mlp.fc1.", ".mlp.net.0.proj.")
        elif ".mlp.fc2." in key:
            new_key = key.replace(".mlp.fc2.", ".mlp.net.2.")
        elif ".attn.qkv." in key:
            if not isinstance(value, torch.Tensor) or value.shape[0] % 3 != 0:
                raise ValueError(f"Cannot split malformed QKV tensor for `{key}` with shape {getattr(value, 'shape', None)}.")
            query, key_tensor, value_tensor = value.chunk(3, dim=0)
            translated[key.replace(".attn.qkv.", ".attn.to_q.")] = query
            translated[key.replace(".attn.qkv.", ".attn.to_k.")] = key_tensor
            translated[key.replace(".attn.qkv.", ".attn.to_v.")] = value_tensor
            continue
        elif ".attn.proj." in key:
            new_key = key.replace(".attn.proj.", ".attn.to_out.0.")
        else:
            new_key = key

        translated[new_key] = value

    return translated


def convert_transformer_state_dict(
    transformer_config: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_key: str | None,
    prefer_ema: bool,
    output_dir: Path,
    safe_serialization: bool,
    verify_load: bool,
    component_name: str,
) -> dict[str, Any]:
    raw_checkpoint = load_checkpoint(checkpoint_path)
    state_dict = unwrap_state_dict(raw_checkpoint, checkpoint_key=checkpoint_key, prefer_ema=prefer_ema)
    state_dict = translate_transformer_state_dict(state_dict)

    model = RAEDiT2DModel(**transformer_config)

    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = set(load_result.missing_keys)
    unexpected_keys = set(load_result.unexpected_keys)

    allowed_missing = {
        "pos_embed",
        "enc_feat_rope.freqs_cos",
        "enc_feat_rope.freqs_sin",
        "dec_feat_rope.freqs_cos",
        "dec_feat_rope.freqs_sin",
    }
    missing_keys -= allowed_missing

    if unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys while converting {component_name}: {sorted(unexpected_keys)[:20]}"
            + (" ..." if len(unexpected_keys) > 20 else "")
        )
    if missing_keys:
        raise RuntimeError(
            f"Missing keys while converting {component_name}: {sorted(missing_keys)[:20]}"
            + (" ..." if len(missing_keys) > 20 else "")
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_config(output_dir)
    weights_path = output_dir / (SAFETENSORS_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME)
    state_dict_to_save = model.state_dict()
    if safe_serialization:
        import safetensors.torch

        safetensors.torch.save_file(state_dict_to_save, weights_path, metadata={"format": "pt"})
    else:
        torch.save(state_dict_to_save, weights_path)

    if verify_load:
        reloaded = RAEDiT2DModel.from_pretrained(output_dir, low_cpu_mem_usage=False)
        if not isinstance(reloaded, RAEDiT2DModel):
            raise RuntimeError(f"Verification failed for {component_name}: reloaded object is not RAEDiT2DModel.")

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_key": checkpoint_key,
        "prefer_ema": prefer_ema,
        "config": transformer_config,
        "num_parameters": sum(t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor)),
    }


def write_metadata(output_path: Path, metadata: dict[str, Any]) -> None:
    with (output_path / "conversion_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)


def resolve_input_path(accessor: RepoAccessor, path: str) -> Path:
    expanded_path = Path(path).expanduser()
    if expanded_path.is_absolute():
        if expanded_path.is_file():
            return expanded_path
        raise FileNotFoundError(f"Absolute path does not exist: {expanded_path}")

    candidates = [path]
    if path.startswith("models/"):
        candidates.append(path[len("models/") :])

    for candidate in candidates:
        try:
            return accessor.fetch(candidate)
        except FileNotFoundError:
            continue

    for candidate in candidates:
        local_path = Path(candidate).expanduser()
        if local_path.is_file():
            return local_path

    raise FileNotFoundError(f"Could not resolve `{path}` from `{accessor.repo_or_path}`.")


def resolve_checkpoint_path(
    accessor: RepoAccessor,
    configured_path: str | None,
    override_path: str | None,
    description: str,
) -> Path | None:
    path = override_path or configured_path
    if path is None:
        return None
    try:
        return resolve_input_path(accessor, path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{description} not found: {path}") from error


def convert(args: argparse.Namespace) -> None:
    weights_accessor = RepoAccessor(args.repo_or_path, cache_dir=args.cache_dir)
    config_accessor = (
        RepoAccessor(args.config_repo_or_path, cache_dir=args.cache_dir)
        if args.config_repo_or_path
        else weights_accessor
    )
    config = read_yaml(config_accessor, args.config_path)

    stage2 = _resolve_section(config, "stage_2", "stage2")
    stage2_params = stage2.get("params", {})
    misc = _resolve_section(config, "misc")
    guidance = _resolve_section(config, "guidance")

    transformer_config = build_transformer_config(stage2_params, misc)
    checkpoint_path = resolve_checkpoint_path(
        weights_accessor,
        configured_path=stage2.get("ckpt"),
        override_path=args.checkpoint_path,
        description="Stage-2 checkpoint",
    )
    if checkpoint_path is None:
        raise ValueError(
            "Could not resolve a Stage-2 checkpoint. Pass `--checkpoint_path` or provide `stage_2.ckpt` in config."
        )

    scheduler, scheduler_metadata = build_scheduler_config(config)
    sampler = _resolve_section(config, "sampler")

    metadata = {
        "source": {
            "weights_repo_or_path": args.repo_or_path,
            "config_repo_or_path": args.config_repo_or_path,
            "config_path": args.config_path,
            "vae_model_name_or_path": args.vae_model_name_or_path,
        },
        "scheduler": scheduler_metadata,
        "sampler": {
            "mode": sampler.get("mode", "ODE"),
            "params": sampler.get("params", {}),
        },
        "guidance": {
            "method": guidance.get("method", "cfg"),
            "scale": float(guidance.get("scale", 1.0)),
            "t_min": float(guidance.get("t-min", guidance.get("t_min", 0.0))),
            "t_max": float(guidance.get("t-max", guidance.get("t_max", 1.0))),
        },
        "misc": misc,
    }

    print(f"Using config:              {args.config_path}")
    print(f"Using Stage-2 checkpoint:  {checkpoint_path}")
    print(f"Derived scheduler shift:   {scheduler.config.shift:.6f}")
    if (
        metadata["sampler"]["mode"] != "ODE"
        or metadata["sampler"]["params"].get("sampling_method", "euler") != "euler"
    ):
        print(
            "Warning: upstream sampler is not the public ODE/Euler path. The saved scheduler still uses "
            "FlowMatchEulerDiscreteScheduler for diffusers V1 compatibility."
        )
    if guidance.get("guidance_model") is not None:
        print("Note: upstream `guidance.guidance_model` is ignored in this diffusers V1 converter.")

    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        print(json.dumps({"transformer_config": transformer_config}, indent=2))
        return

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    transformer_output_dir = output_path / args.transformer_subfolder
    metadata["transformer"] = convert_transformer_state_dict(
        transformer_config=transformer_config,
        checkpoint_path=checkpoint_path,
        checkpoint_key=args.checkpoint_key,
        prefer_ema=not args.disable_ema,
        output_dir=transformer_output_dir,
        safe_serialization=args.safe_serialization,
        verify_load=args.verify_load,
        component_name="transformer",
    )

    scheduler_output_dir = output_path / args.scheduler_subfolder
    scheduler.save_pretrained(scheduler_output_dir)

    if args.vae_model_name_or_path is not None:
        vae = AutoencoderRAE.from_pretrained(args.vae_model_name_or_path, cache_dir=args.cache_dir)
        transformer = RAEDiT2DModel.from_pretrained(transformer_output_dir, low_cpu_mem_usage=False)
        scheduler_for_pipe = FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_output_dir)

        id2label = None
        if args.id2label_json_path is not None:
            with Path(args.id2label_json_path).expanduser().open("r", encoding="utf-8") as handle:
                id2label = json.load(handle)

        pipe = RAEDiTPipeline(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler_for_pipe,
            id2label=id2label,
        )
        pipe.save_pretrained(output_path, safe_serialization=args.safe_serialization)
        metadata["pipeline"] = {"saved": True, "id2label_json_path": args.id2label_json_path}

    write_metadata(output_path, metadata)

    print(f"Saved transformer to:      {transformer_output_dir}")
    print(f"Saved scheduler to:        {scheduler_output_dir}")
    if "pipeline" in metadata:
        print(f"Saved pipeline to:         {output_path}")
    print(f"Saved metadata to:         {output_path / 'conversion_metadata.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert upstream RAE Stage-2 checkpoints to diffusers format")
    parser.add_argument(
        "--repo_or_path",
        type=str,
        required=True,
        help="Hub repo id or local directory containing the upstream Stage-2 weights.",
    )
    parser.add_argument(
        "--config_repo_or_path",
        type=str,
        default=None,
        help="Optional separate hub repo id or local directory containing the upstream YAML configs. Defaults to `repo_or_path`.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Relative path to the upstream Stage-2 YAML config inside `config_repo_or_path`, or a direct local file path.",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save converted components.")
    parser.add_argument(
        "--vae_model_name_or_path",
        type=str,
        default=None,
        help="Optional diffusers AutoencoderRAE checkpoint to bundle into a full RAEDiTPipeline export.",
    )
    parser.add_argument(
        "--id2label_json_path",
        type=str,
        default=None,
        help="Optional JSON mapping of class ids to label strings for the saved pipeline.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional Stage-2 checkpoint override. Interpreted relative to `repo_or_path` unless it is already local.",
    )
    parser.add_argument(
        "--checkpoint_key",
        type=str,
        default=None,
        help="Optional dotted key path inside the Stage-2 checkpoint payload. By default the converter auto-prefers `ema` then `model`.",
    )
    parser.add_argument(
        "--transformer_subfolder",
        type=str,
        default=DEFAULT_TRANSFORMER_SUBFOLDER,
        help="Subfolder name used for the converted primary transformer.",
    )
    parser.add_argument(
        "--scheduler_subfolder",
        type=str,
        default=DEFAULT_SCHEDULER_SUBFOLDER,
        help="Subfolder name used for the saved scheduler config.",
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional Hugging Face Hub cache directory.")
    parser.add_argument(
        "--disable_ema",
        action="store_true",
        help="Do not prefer `ema` when the checkpoint stores both `ema` and `model` weights.",
    )
    parser.add_argument(
        "--safe_serialization",
        action="store_true",
        help="Save converted transformer weights as safetensors.",
    )
    parser.add_argument(
        "--verify_load",
        action="store_true",
        help="Reload each saved transformer with `from_pretrained(low_cpu_mem_usage=False)` after conversion.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only resolve paths and print the derived config/metadata without saving anything.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()
