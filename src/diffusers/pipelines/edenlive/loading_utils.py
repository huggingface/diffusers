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

"""Read the four original EdenLive releases without writing converted checkpoints."""

import copy
import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path, PurePosixPath

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
    DINOv3ViTConfig,
    DINOv3ViTModel,
    UMT5EncoderModel,
)

from ...loaders.single_file_utils import convert_wan_vae_to_diffusers
from ...models import AutoencoderKLWan, EdenLiveLightTAEDecoder, EdenLiveProxyProjection, EdenLiveTransformer3DModel
from ...models.model_loading_utils import load_state_dict
from ...schedulers import EdenLiveFlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from ...utils import _resolve_dtype
from ...utils.hub_utils import _resolve_revision


_EDENLIVE_PROFILES = {
    "5b-bidir": {"size": "5b", "causal": False, "transformer": "dit-5b-bidir", "resize": "bilinear"},
    "14b-bidir": {"size": "14b", "causal": False, "transformer": "dit-14b-bidir", "resize": "bicubic"},
    "5b-causal": {"size": "5b", "causal": True, "transformer": "dit-5b", "resize": "bilinear"},
    "14b-causal": {"size": "14b", "causal": True, "transformer": "dit-14b", "resize": "bicubic_center_crop"},
}

_COMPONENT_CLASSES = {
    "transformer": ("diffusers", "EdenLiveTransformer3DModel"),
    "vae": ("diffusers", "AutoencoderKLWan"),
    "decoder": ("diffusers", "EdenLiveLightTAEDecoder"),
    "text_encoder": ("transformers", "UMT5EncoderModel"),
    "tokenizer": ("transformers", "AutoTokenizer"),
    "proxy_encoder": ("transformers", "DINOv3ViTModel"),
    "proxy_projection": ("diffusers", "EdenLiveProxyProjection"),
    "image_encoder": ("transformers", "CLIPVisionModel"),
    "image_processor": ("transformers", "CLIPImageProcessor"),
}

_HUB_OPTIONS = {"cache_dir", "force_download", "proxies", "local_files_only", "token", "revision"}

_UNSUPPORTED_LOADING_OPTIONS = {
    "device_map",
    "max_memory",
    "offload_folder",
    "offload_state_dict",
    "variant",
    "quantization_config",
    "custom_pipeline",
    "custom_revision",
    "use_onnx",
    "use_flashpack",
    "load_connected_pipeline",
    "provider",
    "sess_options",
    "provider_options",
}


def read_checkpoint_config(source, loading_options):
    options = loading_options.copy()
    if Path(source).is_dir():
        config_file = Path(source) / "model_index.json"
    else:
        options["revision"] = _resolve_revision(
            source,
            **{
                key: value
                for key, value in options.items()
                if key in {"revision", "cache_dir", "local_files_only", "token"}
            },
        )
        config_file = Path(
            hf_hub_download(
                str(source),
                "model_index.json",
                **{key: value for key, value in options.items() if key in _HUB_OPTIONS},
            )
        )
    with config_file.open() as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("EdenLive model_index.json must contain an object.")
    return config, config_file, options


def normalize_saved_profile(config, requested_profile):
    profile = config.get("profile")
    if profile in ("5b", "14b"):
        modes = {"EdenLivePipeline": "bidir", "EdenLiveChunkCausalPipeline": "causal"}
        mode = modes.get(config.get("_class_name"))
        if mode is None:
            raise ValueError("Cannot infer an EdenLive profile from this legacy pipeline class.")
        profile = f"{profile}-{mode}"
    if profile not in _EDENLIVE_PROFILES:
        raise ValueError(f"Saved EdenLive pipeline has an invalid profile: {profile!r}.")
    if requested_profile is not None and requested_profile != profile:
        raise ValueError(f"Requested profile {requested_profile!r} conflicts with saved profile {profile!r}.")
    return profile


def load_original_pipeline(pipeline_class, source, config_file, config, profile, loading_options):
    component_names, option_names = pipeline_class._get_signature_keys(pipeline_class)
    passed_components = {name: loading_options[name] for name in component_names if name in loading_options}
    context = resolve_original_checkpoint(
        config, profile=profile, passed_components=passed_components, loading_options=loading_options
    )
    supported_options = (
        _HUB_OPTIONS
        | _UNSUPPORTED_LOADING_OPTIONS
        | {
            "dtype",
            "torch_dtype",
            "low_cpu_mem_usage",
            "disable_mmap",
            "use_safetensors",
            "trust_remote_code",
        }
    )
    unknown = set(loading_options) - set(component_names) - set(option_names) - supported_options
    if unknown:
        raise ValueError(f"Original EdenLive loading does not support {', '.join(sorted(unknown))}.")
    dtype = _resolve_dtype(loading_options.get("dtype"), loading_options.get("torch_dtype"))
    if dtype is None:
        dtype = context["default_dtype"]
    if not isinstance(dtype, (dict, torch.dtype)) or (
        isinstance(dtype, dict) and any(not isinstance(value, torch.dtype) for value in dtype.values())
    ):
        raise ValueError("`dtype` must be a torch.dtype or a mapping of component names to torch.dtype values.")
    folder = config_file.parent
    if not Path(source).is_dir() and (
        loading_options.get("force_download", False)
        or not all((folder / filename).is_file() for filename in context["files"])
    ):
        folder = Path(
            snapshot_download(
                str(source),
                allow_patterns=context["files"],
                user_agent={"pipeline_class": pipeline_class.__name__},
                **{key: value for key, value in loading_options.items() if key in _HUB_OPTIONS},
            )
        )
    missing = [filename for filename in context["files"] if not (folder / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"Original checkpoint files are missing from {folder}: {missing}")
    init_kwargs = {name: context["config"][name] for name in option_names if name in context["config"]}
    init_kwargs.update({name: loading_options[name] for name in option_names if name in loading_options})
    try:
        for name in component_names:
            if name in passed_components:
                init_kwargs[name] = passed_components[name]
            elif context["config"][name][0] is None:
                init_kwargs[name] = None
            else:
                component_dtype = (
                    dtype.get(name, dtype.get("default", torch.float32)) if isinstance(dtype, dict) else dtype
                )
                init_kwargs[name] = load_original_component(
                    name, cached_folder=folder, context=context, dtype=component_dtype, loading_options=loading_options
                )
        pipeline = pipeline_class(**init_kwargs)
    finally:
        context["state_dicts"].clear()
    pipeline.register_to_config(_name_or_path=str(source))
    return pipeline


def _relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or any(c in value for c in "*?["):
        raise ValueError("EdenLive path overrides must be literal repository-relative paths.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"EdenLive path must stay within the checkpoint repository: {value!r}.")
    return path.as_posix()


def _component_sources(profile_name, path_overrides):
    profile = _EDENLIVE_PROFILES[profile_name]
    size, transformer = profile["size"], profile["transformer"]
    sources = {
        "transformer": {
            "config": f"{transformer}/config.json",
            "weights": f"{transformer}/diffusion_pytorch_model.safetensors",
        },
        "vae": {
            "config": f"vae-{size}/config.json",
            "weights": f"vae-{size}/Wan2.{'2' if size == '5b' else '1'}_VAE.pth",
        },
        "decoder": {"weights": f"decoder-{size}/lighttaew2_{'2' if size == '5b' else '1'}.pth"},
        "text_encoder": {"config": "umt5/text_encoder/config.json", "weights": "umt5/text_encoder/model.safetensors"},
        "tokenizer": {"directory": "umt5/tokenizer"},
        "proxy_encoder": {
            "config": "vision-encoder/backbone/config.json",
            "weights": "vision-encoder/projection.safetensors",
        },
        "proxy_projection": {"weights": "vision-encoder/projection.safetensors"},
        "image_encoder": {
            "config": "clip/image_encoder/config.json",
            "weights": "clip/image_encoder/model.safetensors",
        },
        "image_processor": {"directory": "clip/image_processor"},
    }
    if not isinstance(path_overrides, dict) or set(path_overrides) - set(sources):
        raise ValueError("`_path_overrides` must map EdenLive component names to paths.")
    for name, override in path_overrides.items():
        if not isinstance(override, dict):
            raise ValueError(f"Path overrides for {name} must be an object.")
        # Only transformers vary by full profile; VAE and decoder files vary by size.
        selectors = (
            set(_EDENLIVE_PROFILES)
            if name == "transformer"
            else {"5b", "14b"}
            if name in ("vae", "decoder")
            else set()
        )
        if selectors and set(override) <= selectors:
            selector = profile_name if name == "transformer" else size
            for value in override.values():
                if not isinstance(value, dict) or set(value) - (set(sources[name]) | {"directory"}):
                    raise ValueError(f"Invalid path fields for {name}.")
                for path in value.values():
                    _relative_path(path)
            override = override.get(selector, {})
        if set(override) - (set(sources[name]) | {"directory"}):
            raise ValueError(f"Invalid path fields for {name}: {sorted(override)}.")
        override = {key: _relative_path(value) for key, value in override.items()}
        if "directory" in override and "directory" not in sources[name]:
            directory = override.pop("directory")
            sources[name] = {key: f"{directory}/{PurePosixPath(value).name}" for key, value in sources[name].items()}
        sources[name].update(override)
    if size == "5b":
        sources.pop("image_encoder")
        sources.pop("image_processor")
    return sources


def _source_files(name, source):
    if name == "tokenizer":
        return [
            f"{source['directory']}/{filename}"
            for filename in ("tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "spiece.model")
        ]
    if name == "image_processor":
        return [f"{source['directory']}/preprocessor_config.json"]
    return list(source.values())


def _read_config(folder, source):
    with open(Path(folder) / source["config"]) as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected an object in {source['config']}.")
    return config


def _read_component_state(folder, name, context, disable_mmap):
    filename = context["sources"][name]["weights"]
    state = context["state_dicts"].get(filename)
    if state is None:
        state = load_state_dict(str(Path(folder) / filename), disable_mmap=disable_mmap, map_location="cpu")
        if not isinstance(state, dict) or any(
            not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()
        ):
            raise ValueError(f"Expected a tensor state dictionary in {filename}.")
    context["source_users"][filename] -= 1
    if context["source_users"][filename] > 0:
        context["state_dicts"][filename] = state
    else:
        context["state_dicts"].pop(filename, None)
    return state


def _strict_assign(model, state, dtype):
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(dtype=dtype).eval().requires_grad_(False)


def _empty_weights(loading_options):
    return torch.device("meta") if loading_options.get("low_cpu_mem_usage", True) else nullcontext()


def _proxy_state(state):
    normalized = {}
    for key, value in state.items():
        for prefix in ("model.module.", "model.", "module."):
            if key.startswith(prefix):
                key = key.removeprefix(prefix)
                break
        key = key.removeprefix("encoder.")
        if key in normalized:
            raise ValueError(f"Duplicate proxy checkpoint key: {key}.")
        if not key.startswith(("backbone.", "patch_projection.")) and key not in (
            "global_projection.weight",
            "global_projection.bias",
        ):
            raise ValueError(f"Unrecognized proxy checkpoint key: {key}.")
        normalized[key] = value
    return normalized


def _load_transformers_component(model_class, config, state, dtype, loading_options, allowed_unused=()):
    state = {key: value for key, value in state.items() if key not in allowed_unused}
    model, info = model_class.from_pretrained(
        None,
        config=model_class.config_class.from_dict(config),
        state_dict=state,
        dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=loading_options.get("low_cpu_mem_usage", True),
        output_loading_info=True,
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"{model_class.__name__} checkpoint did not load strictly: {info}.")
    return model.eval().requires_grad_(False)


def _remap_wan22_vae_key(key):
    """Normalize only the nested Wan 2.2 blocks missing from the native mapper."""
    parts = key.split(".")
    if len(parts) < 5 or parts[:2] not in (["encoder", "downsamples"], ["decoder", "upsamples"]):
        return key
    block = "down_blocks" if parts[0] == "encoder" else "up_blocks"
    prefix = f"{parts[0]}.{block}.{parts[2]}"
    if parts[3] == "avg_shortcut":
        return f"{prefix}.avg_shortcut.{'.'.join(parts[4:])}"
    if len(parts) < 6 or parts[3] != parts[1]:
        return key
    slot, rest = parts[4], parts[5:]
    if rest[0] == "residual" and len(rest) > 2:
        field = {"0": "norm1", "2": "conv1", "3": "norm2", "6": "conv2"}.get(rest[1])
        if field is not None:
            return f"{prefix}.resnets.{slot}.{field}.{'.'.join(rest[2:])}"
    elif rest[0] == "shortcut":
        return f"{prefix}.resnets.{slot}.conv_shortcut.{'.'.join(rest[1:])}"
    elif parts[0] == "encoder" and slot == "2":
        return f"{prefix}.downsampler.{'.'.join(rest)}"
    elif parts[0] == "decoder" and slot == "3":
        return f"{prefix}.upsampler.{'.'.join(rest)}"
    return key


def resolve_original_checkpoint(config_dict, *, profile, passed_components, loading_options):
    if "_edenlive_original" not in config_dict:
        return None
    original = config_dict["_edenlive_original"]
    if not isinstance(original, dict) or set(original) - {"_path_overrides"}:
        raise ValueError("`_edenlive_original` must be an object with optional `_path_overrides`.")
    if profile not in _EDENLIVE_PROFILES:
        raise ValueError(
            f"Original EdenLive checkpoints require `profile` in {tuple(_EDENLIVE_PROFILES)}; got {profile!r}."
        )
    preset = _EDENLIVE_PROFILES[profile]
    unsupported = {
        key: loading_options.get(key)
        for key in _UNSUPPORTED_LOADING_OPTIONS
        if loading_options.get(key) is not None and loading_options.get(key) is not False
    }
    if unsupported:
        raise ValueError(
            f"Original EdenLive loading does not support {', '.join(sorted(unsupported))}; load first and place the pipeline explicitly."
        )
    sources = _component_sources(profile, original.get("_path_overrides", {}))
    sources = {name: source for name, source in sources.items() if name not in passed_components}
    safe_only = loading_options.get("use_safetensors")
    weights = [source["weights"] for source in sources.values() if "weights" in source]
    if safe_only is True and any(not path.endswith(".safetensors") for path in weights):
        raise ValueError(
            "Original EdenLive includes PyTorch .pth components; omit `use_safetensors=True` or provide those components explicitly."
        )
    if safe_only is False and any(path.endswith(".safetensors") for path in weights):
        raise ValueError(
            "Original EdenLive includes Safetensors components; omit `use_safetensors=False` or provide those components explicitly."
        )
    config = copy.deepcopy(config_dict)
    config.pop("_edenlive_original")
    config.update(
        _class_name="EdenLivePipeline",
        profile=profile,
        proxy_resize_mode=preset["resize"],
        proxy_unconditional="zero" if profile == "14b-causal" else "same",
        proxy_projection_autocast=True,
        proxy_image_size=(480, 832),
    )
    config.update(_COMPONENT_CLASSES)
    if preset["size"] == "5b":
        config.update(image_encoder=(None, None), image_processor=(None, None))
    scheduler_name = "UniPCMultistepScheduler" if profile == "5b-bidir" else "EdenLiveFlowMatchEulerDiscreteScheduler"
    config["scheduler"] = ("diffusers", scheduler_name)
    if preset["causal"]:
        config.update(chunk_size=3, max_context_frames=18, sink_frames=6)
    return {
        "config": config,
        "profile": profile,
        "preset": preset,
        "sources": sources,
        "files": sorted(
            {"model_index.json"}
            | {filename for name, source in sources.items() for filename in _source_files(name, source)}
        ),
        "default_dtype": {
            "default": torch.bfloat16,
            "proxy_encoder": torch.float32,
            "proxy_projection": torch.float32,
        },
        "state_dicts": {},
        "source_users": Counter(source["weights"] for source in sources.values() if "weights" in source),
    }


def load_original_component(name, *, cached_folder, context, dtype, loading_options):
    preset = context["preset"]
    if name == "scheduler":
        if context["profile"] == "5b-bidir":
            return UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=5.0)
        if preset["causal"]:
            timesteps = [1000, 937.5, 833.3333333333, 625] if preset["size"] == "5b" else [1000, 900, 750, 500]
            return EdenLiveFlowMatchEulerDiscreteScheduler(stochastic_sampling=True, default_timesteps=timesteps)
        return EdenLiveFlowMatchEulerDiscreteScheduler(shift=3.0)
    source = context["sources"][name]
    if name in ("tokenizer", "image_processor"):
        model_class = AutoTokenizer if name == "tokenizer" else CLIPImageProcessor
        return model_class.from_pretrained(Path(cached_folder) / source["directory"], local_files_only=True)
    state = _read_component_state(cached_folder, name, context, loading_options.get("disable_mmap", False))
    config = _read_config(cached_folder, source) if "config" in source else None
    if name == "transformer":
        proxy_channels = state["patch_embedding.weight"].shape[1] - config["in_channels"]
        expected_proxy_channels = config.get("proxy_channels", 1024)
        condition_rmsnorm = "patch_embedding.condition_norm.weight" in state
        if proxy_channels != expected_proxy_channels or condition_rmsnorm != config.get(
            "condition_rmsnorm", preset["size"] == "14b"
        ):
            raise ValueError(
                "Transformer proxy channels or normalization do not match its configuration and selected profile."
            )
        with _empty_weights(loading_options):
            model = EdenLiveTransformer3DModel.from_config(
                config, proxy_channels=proxy_channels, condition_rmsnorm=condition_rmsnorm
            )
        # These rotary buffers are derived, not serialized checkpoint tensors.
        for attribute in ("rope", "cache_rope"):
            rope = getattr(model, attribute)
            setattr(model, attribute, type(rope)(rope.attention_head_dim, tuple(rope.patch_size), rope.max_seq_len))
        return _strict_assign(model, state, dtype)
    if name == "vae":
        mapped = {}
        for key, value in state.items():
            target = _remap_wan22_vae_key(key)
            if target in mapped:
                raise ValueError(f"Multiple VAE keys map to {target}.")
            mapped[target] = value
        mapped = convert_wan_vae_to_diffusers(mapped)
        if len(mapped) != len(state):
            raise ValueError("Multiple VAE checkpoint keys map to the same parameter.")
        # Wan 2.2 stores folded RGB convolutions; the current native class
        # takes their post-patchify/pre-unpatchify channel counts.
        patch_size = config.get("patch_size") or 1
        for field, key, dimension in (
            ("in_channels", "encoder.conv_in.weight", 1),
            ("out_channels", "decoder.conv_out.weight", 0),
        ):
            channels = mapped[key].shape[dimension]
            declared = config.get(field, 3)
            if channels not in (declared, declared * patch_size**2):
                raise ValueError(f"VAE {field} does not match the checkpoint convolutions.")
            config[field] = channels
        with _empty_weights(loading_options):
            model = AutoencoderKLWan.from_config(config)
        return _strict_assign(model, mapped, dtype)
    if name == "decoder":
        with _empty_weights(loading_options):
            model = EdenLiveLightTAEDecoder(
                latent_channels=48 if preset["size"] == "5b" else 16, patch_size=2 if preset["size"] == "5b" else 1
            )
        return _strict_assign(model, model._convert_original_state_dict(state), dtype)
    if name in ("proxy_encoder", "proxy_projection"):
        state = _proxy_state(state)
        if name == "proxy_encoder":
            config = DINOv3ViTConfig.from_dict(config)
            with _empty_weights(loading_options):
                model = DINOv3ViTModel(config)
            expected = set(model.state_dict())
            mapped = {}
            for key, value in state.items():
                if key.startswith("backbone."):
                    target = key.removeprefix("backbone.")
                    if target not in expected and f"model.{target}" in expected:
                        target = f"model.{target}"
                    mapped[target] = value
            model.rope_embeddings = type(model.rope_embeddings)(config)
        else:
            mapped = {key: value for key, value in state.items() if key.startswith("patch_projection.")}
            weight = mapped["patch_projection.weight"]
            with _empty_weights(loading_options):
                model = EdenLiveProxyProjection(in_channels=weight.shape[1], out_channels=weight.shape[0])
        return _strict_assign(model, mapped, dtype)
    if name == "text_encoder":
        return _load_transformers_component(UMT5EncoderModel, config, state, dtype, loading_options)
    if name == "image_encoder":
        return _load_transformers_component(
            CLIPVisionModel, config, state, dtype, loading_options, ("visual_projection.weight",)
        )
    raise ValueError(f"Unknown EdenLive component: {name}.")
