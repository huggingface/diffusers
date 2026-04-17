import argparse
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download

from diffusers import AutoencoderRAE


DECODER_CONFIGS = {
    "ViTB": {
        "decoder_hidden_size": 768,
        "decoder_intermediate_size": 3072,
        "decoder_num_attention_heads": 12,
        "decoder_num_hidden_layers": 12,
    },
    "ViTL": {
        "decoder_hidden_size": 1024,
        "decoder_intermediate_size": 4096,
        "decoder_num_attention_heads": 16,
        "decoder_num_hidden_layers": 24,
    },
    "ViTXL": {
        "decoder_hidden_size": 1152,
        "decoder_intermediate_size": 4096,
        "decoder_num_attention_heads": 16,
        "decoder_num_hidden_layers": 28,
    },
}

ENCODER_DEFAULT_NAME_OR_PATH = {
    "dinov2": "facebook/dinov2-with-registers-base",
    "siglip2": "google/siglip2-base-patch16-256",
    "mae": "facebook/vit-mae-base",
}

ENCODER_HIDDEN_SIZE = {
    "dinov2": 768,
    "siglip2": 768,
    "mae": 768,
}

ENCODER_PATCH_SIZE = {
    "dinov2": 14,
    "siglip2": 16,
    "mae": 16,
}

DEFAULT_DECODER_SUBDIR = {
    "dinov2": "decoders/dinov2/wReg_base",
    "mae": "decoders/mae/base_p16",
    "siglip2": "decoders/siglip2/base_p16_i256",
}

DEFAULT_STATS_SUBDIR = {
    "dinov2": "stats/dinov2/wReg_base",
    "mae": "stats/mae/base_p16",
    "siglip2": "stats/siglip2/base_p16_i256",
}

DECODER_FILE_CANDIDATES = ("dinov2_decoder.pt", "model.pt")
STATS_FILE_CANDIDATES = ("stat.pt",)


def dataset_case_candidates(name: str) -> tuple[str, ...]:
    return (name, name.lower(), name.upper(), name.title(), "imagenet1k", "ImageNet1k")


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
            return self.local_root / relative_path
        downloaded = hf_hub_download(repo_id=self.repo_id, filename=relative_path, cache_dir=self.cache_dir)
        return Path(downloaded)


def unwrap_state_dict(maybe_wrapped: dict[str, Any]) -> dict[str, Any]:
    state_dict = maybe_wrapped
    for k in ("model", "module", "state_dict"):
        if isinstance(state_dict, dict) and k in state_dict and isinstance(state_dict[k], dict):
            state_dict = state_dict[k]

    out = dict(state_dict)
    if len(out) > 0 and all(key.startswith("module.") for key in out):
        out = {key[len("module.") :]: value for key, value in out.items()}
    if len(out) > 0 and all(key.startswith("decoder.") for key in out):
        out = {key[len("decoder.") :]: value for key, value in out.items()}
    return out


def remap_decoder_attention_keys_for_diffusers(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Map official RAE decoder attention key layout to diffusers Attention layout used by AutoencoderRAE decoder.

    Example mappings:
    - `...attention.attention.query.*` -> `...attention.to_q.*`
    - `...attention.attention.key.*`   -> `...attention.to_k.*`
    - `...attention.attention.value.*` -> `...attention.to_v.*`
    - `...attention.output.dense.*`    -> `...attention.to_out.0.*`
    """
    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace(".attention.attention.query.", ".attention.to_q.")
        new_key = new_key.replace(".attention.attention.key.", ".attention.to_k.")
        new_key = new_key.replace(".attention.attention.value.", ".attention.to_v.")
        new_key = new_key.replace(".attention.output.dense.", ".attention.to_out.0.")
        remapped[new_key] = value
    return remapped


def resolve_decoder_file(
    accessor: RepoAccessor, encoder_type: str, variant: str, decoder_checkpoint: str | None
) -> str:
    if decoder_checkpoint is not None:
        if accessor.exists(decoder_checkpoint):
            return decoder_checkpoint
        raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_checkpoint}")

    base = f"{DEFAULT_DECODER_SUBDIR[encoder_type]}/{variant}"
    for name in DECODER_FILE_CANDIDATES:
        candidate = f"{base}/{name}"
        if accessor.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find decoder checkpoint under `{base}`. Tried: {list(DECODER_FILE_CANDIDATES)}"
    )


def resolve_stats_file(
    accessor: RepoAccessor,
    encoder_type: str,
    dataset_name: str,
    stats_checkpoint: str | None,
) -> str | None:
    if stats_checkpoint is not None:
        if accessor.exists(stats_checkpoint):
            return stats_checkpoint
        raise FileNotFoundError(f"Stats checkpoint not found: {stats_checkpoint}")

    base = DEFAULT_STATS_SUBDIR[encoder_type]
    for dataset in dataset_case_candidates(dataset_name):
        for name in STATS_FILE_CANDIDATES:
            candidate = f"{base}/{dataset}/{name}"
            if accessor.exists(candidate):
                return candidate

    return None


def extract_latent_stats(stats_obj: Any) -> tuple[Any | None, Any | None]:
    if not isinstance(stats_obj, dict):
        return None, None

    if "latents_mean" in stats_obj or "latents_std" in stats_obj:
        return stats_obj.get("latents_mean", None), stats_obj.get("latents_std", None)

    mean = stats_obj.get("mean", None)
    var = stats_obj.get("var", None)
    if mean is None and var is None:
        return None, None

    latents_std = None
    if var is not None:
        if isinstance(var, torch.Tensor):
            latents_std = torch.sqrt(var + 1e-5)
        else:
            latents_std = torch.sqrt(torch.tensor(var) + 1e-5)
    return mean, latents_std


def _strip_final_layernorm_affine(state_dict: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Remove final layernorm weight/bias from encoder state dict.

    RAE uses non-affine layernorm (weight=1, bias=0 is the default identity).
    Stripping these keys means the model keeps its default init values, which
    is functionally equivalent to setting elementwise_affine=False.
    """
    keys_to_strip = {f"{prefix}weight", f"{prefix}bias"}
    return {k: v for k, v in state_dict.items() if k not in keys_to_strip}


def _load_hf_encoder_state_dict(encoder_type: str, encoder_name_or_path: str) -> dict[str, Any]:
    """Download the HF encoder and extract the state dict for the inner model."""
    if encoder_type == "dinov2":
        from transformers import Dinov2WithRegistersModel

        hf_model = Dinov2WithRegistersModel.from_pretrained(encoder_name_or_path)
        sd = hf_model.state_dict()
        return _strip_final_layernorm_affine(sd, prefix="layernorm.")
    elif encoder_type == "siglip2":
        from transformers import SiglipModel

        # SiglipModel.vision_model is a SiglipVisionTransformer.
        # Our Siglip2Encoder wraps it inside SiglipVisionModel which nests it
        # under .vision_model, so we add the prefix to match the diffusers key layout.
        hf_model = SiglipModel.from_pretrained(encoder_name_or_path).vision_model
        sd = {f"vision_model.{k}": v for k, v in hf_model.state_dict().items()}
        return _strip_final_layernorm_affine(sd, prefix="vision_model.post_layernorm.")
    elif encoder_type == "mae":
        from transformers import ViTMAEForPreTraining

        hf_model = ViTMAEForPreTraining.from_pretrained(encoder_name_or_path).vit
        sd = hf_model.state_dict()
        return _strip_final_layernorm_affine(sd, prefix="layernorm.")
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")


def convert(args: argparse.Namespace) -> None:
    accessor = RepoAccessor(args.repo_or_path, cache_dir=args.cache_dir)
    encoder_name_or_path = args.encoder_name_or_path or ENCODER_DEFAULT_NAME_OR_PATH[args.encoder_type]

    decoder_relpath = resolve_decoder_file(accessor, args.encoder_type, args.variant, args.decoder_checkpoint)
    stats_relpath = resolve_stats_file(accessor, args.encoder_type, args.dataset_name, args.stats_checkpoint)

    print(f"Using decoder checkpoint: {decoder_relpath}")
    if stats_relpath is not None:
        print(f"Using stats checkpoint:   {stats_relpath}")
    else:
        print("No stats checkpoint found; conversion will proceed without latent stats.")

    if args.dry_run:
        return

    decoder_path = accessor.fetch(decoder_relpath)
    decoder_obj = torch.load(decoder_path, map_location="cpu")
    decoder_state_dict = unwrap_state_dict(decoder_obj)
    decoder_state_dict = remap_decoder_attention_keys_for_diffusers(decoder_state_dict)

    latents_mean, latents_std = None, None
    if stats_relpath is not None:
        stats_path = accessor.fetch(stats_relpath)
        stats_obj = torch.load(stats_path, map_location="cpu")
        latents_mean, latents_std = extract_latent_stats(stats_obj)

    decoder_cfg = DECODER_CONFIGS[args.decoder_config_name]

    # Read encoder normalization stats from the HF image processor (only place that downloads encoder info)
    from transformers import AutoConfig, AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(encoder_name_or_path)
    encoder_norm_mean = list(proc.image_mean)
    encoder_norm_std = list(proc.image_std)

    # Read encoder hidden size and patch size from HF config
    encoder_hidden_size = ENCODER_HIDDEN_SIZE[args.encoder_type]
    encoder_patch_size = ENCODER_PATCH_SIZE[args.encoder_type]
    try:
        hf_config = AutoConfig.from_pretrained(encoder_name_or_path)
        # For models like SigLIP that nest vision config
        if hasattr(hf_config, "vision_config"):
            hf_config = hf_config.vision_config
        encoder_hidden_size = hf_config.hidden_size
        encoder_patch_size = hf_config.patch_size
    except Exception:
        pass

    # Load the actual encoder weights from HF to include in the saved model
    encoder_state_dict = _load_hf_encoder_state_dict(args.encoder_type, encoder_name_or_path)

    # Build model on meta device to avoid double init overhead
    with torch.device("meta"):
        model = AutoencoderRAE(
            encoder_type=args.encoder_type,
            encoder_hidden_size=encoder_hidden_size,
            encoder_patch_size=encoder_patch_size,
            encoder_input_size=args.encoder_input_size,
            patch_size=args.patch_size,
            image_size=args.image_size,
            num_channels=args.num_channels,
            encoder_norm_mean=encoder_norm_mean,
            encoder_norm_std=encoder_norm_std,
            decoder_hidden_size=decoder_cfg["decoder_hidden_size"],
            decoder_num_hidden_layers=decoder_cfg["decoder_num_hidden_layers"],
            decoder_num_attention_heads=decoder_cfg["decoder_num_attention_heads"],
            decoder_intermediate_size=decoder_cfg["decoder_intermediate_size"],
            latents_mean=latents_mean,
            latents_std=latents_std,
            scaling_factor=args.scaling_factor,
        )

    # Assemble full state dict and load with assign=True
    full_state_dict = {}

    # Encoder weights (prefixed with "encoder.")
    for k, v in encoder_state_dict.items():
        full_state_dict[f"encoder.{k}"] = v

    # Decoder weights (prefixed with "decoder.")
    for k, v in decoder_state_dict.items():
        full_state_dict[f"decoder.{k}"] = v

    # Buffers from config
    full_state_dict["encoder_mean"] = torch.tensor(encoder_norm_mean, dtype=torch.float32).view(1, 3, 1, 1)
    full_state_dict["encoder_std"] = torch.tensor(encoder_norm_std, dtype=torch.float32).view(1, 3, 1, 1)
    if latents_mean is not None:
        latents_mean_t = latents_mean if isinstance(latents_mean, torch.Tensor) else torch.tensor(latents_mean)
        full_state_dict["_latents_mean"] = latents_mean_t
    else:
        full_state_dict["_latents_mean"] = torch.zeros(1)
    if latents_std is not None:
        latents_std_t = latents_std if isinstance(latents_std, torch.Tensor) else torch.tensor(latents_std)
        full_state_dict["_latents_std"] = latents_std_t
    else:
        full_state_dict["_latents_std"] = torch.ones(1)

    model.load_state_dict(full_state_dict, strict=False, assign=True)

    # Verify no critical keys are missing
    model_keys = {name for name, _ in model.named_parameters()}
    model_keys |= {name for name, _ in model.named_buffers()}
    loaded_keys = set(full_state_dict.keys())
    missing = model_keys - loaded_keys
    # decoder_pos_embed is initialized in-model. trainable_cls_token is only
    # allowed to be missing if it was absent in the source decoder checkpoint.
    allowed_missing = {"decoder.decoder_pos_embed"}
    if "trainable_cls_token" not in decoder_state_dict:
        allowed_missing.add("decoder.trainable_cls_token")
    if missing - allowed_missing:
        print(f"Warning: missing keys after conversion: {sorted(missing - allowed_missing)}")

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)

    if args.verify_load:
        print("Verifying converted checkpoint with AutoencoderRAE.from_pretrained(low_cpu_mem_usage=False)...")
        loaded_model = AutoencoderRAE.from_pretrained(output_path, low_cpu_mem_usage=False)
        if not isinstance(loaded_model, AutoencoderRAE):
            raise RuntimeError("Verification failed: loaded object is not AutoencoderRAE.")
        print("Verification passed.")

    print(f"Saved converted AutoencoderRAE to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RAE decoder checkpoints to diffusers AutoencoderRAE format")
    parser.add_argument(
        "--repo_or_path", type=str, required=True, help="Hub repo id (e.g. nyu-visionx/RAE-collections) or local path"
    )
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save converted model")

    parser.add_argument("--encoder_type", type=str, choices=["dinov2", "mae", "siglip2"], required=True)
    parser.add_argument(
        "--encoder_name_or_path", type=str, default=None, help="Optional encoder HF model id or local path override"
    )

    parser.add_argument("--variant", type=str, default="ViTXL_n08", help="Decoder variant folder name")
    parser.add_argument("--dataset_name", type=str, default="imagenet1k", help="Stats dataset folder name")

    parser.add_argument(
        "--decoder_checkpoint", type=str, default=None, help="Relative path to decoder checkpoint inside repo/path"
    )
    parser.add_argument(
        "--stats_checkpoint", type=str, default=None, help="Relative path to stats checkpoint inside repo/path"
    )

    parser.add_argument("--decoder_config_name", type=str, choices=list(DECODER_CONFIGS.keys()), default="ViTXL")
    parser.add_argument("--encoder_input_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--scaling_factor", type=float, default=1.0)

    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Only resolve and print selected files")
    parser.add_argument(
        "--verify_load",
        action="store_true",
        help="After conversion, load back with AutoencoderRAE.from_pretrained(low_cpu_mem_usage=False).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()
