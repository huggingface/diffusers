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

"""`diffusers-cli run` — single agentic entry point.

Runs any diffusers pipeline (standard or modular) by forwarding `--pipeline-kwargs` verbatim, saves the output by
detecting its runtime type, and can submit the same call to an HF Sandbox via `--remote`.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import sys
import time
import uuid
import wave
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter, _SubParsersAction
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import torch
from huggingface_hub import HfApi, Sandbox, Volume, get_token, parse_hf_uri
from huggingface_hub.cli._output import out
from huggingface_hub.utils import send_telemetry
from PIL import Image

import diffusers
from diffusers import ContextParallelConfig
from diffusers.models.attention_dispatch import _HUB_KERNELS_REGISTRY
from diffusers.utils import export_to_video, load_image, load_video, logging
from diffusers.utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from diffusers.utils.torch_utils import torch_device

from . import BaseDiffusersCLICommand


logger = logging.get_logger("diffusers-cli/run")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = str(Path.home() / ".diffusers" / "cli" / "run" / "outputs")
DTYPE_CHOICES = ("auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32")
CPU_OFFLOAD_CHOICES = ("model", "group", "auto")


ATTENTION_BACKEND_CHOICES = ("default", *sorted(b.value for b in _HUB_KERNELS_REGISTRY))

# Kwarg keys whose string value gets auto-loaded before being passed to the pipeline call.
# Images resolve via `diffusers.utils.load_image` → PIL.Image.Image; videos resolve via
# `diffusers.utils.load_video` → list[PIL.Image.Image].
_IMAGE_INPUT_KEYS = (
    "image",
    "last_image",
    "mask_image",
    "control_image",
    "ip_adapter_image",
    "image_2",
)
_VIDEO_INPUT_KEYS = (
    "video",
    "control_video",
)
_AUDIO_INPUT_KEYS = (
    "initial_audio_waveforms",
    "reference_audio",
    "src_audio",
)

# Pipeline attribute prefixes that identify a denoiser submodule. Matches base names
# (`transformer`, `unet`) and their numbered variants (`transformer_2`, etc.).
_DENOISER_COMPONENT_KEYS = ("transformer", "unet")

_DEFAULT_REMOTE_DEPS = (
    "diffusers",
    "accelerate",
    "transformers",
    "safetensors",
    "sentencepiece",  # required by several text-encoder tokenizers (T5, LLaMA, …)
    "ftfy",  # required by older CLIP text-encoder paths
    "peft",  # required by `load_lora_weights` when `--lora` is passed
    "imageio",  # preferred `export_to_video` backend
    "imageio-ffmpeg",  # bundles a static ffmpeg; the cv2 fallback needs system libs the slim image lacks
)

# Base sandbox image — provides torch + CUDA so `uv pip install --system`
# only has to add the small Python deps. cuda12.8 is the highest cuda12.x tag
# below the HF Jobs host driver's CUDA 12.9 max.
_DEFAULT_REMOTE_IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"

# Installed console-script name invoked inside the sandbox after the deps land.
_CONTAINER_CLI_BINARY = "diffusers-cli"

# Working directories inside the sandbox: local media from `--pipeline-kwargs` is uploaded
# under _SANDBOX_INPUTS_DIR, and the sandbox CLI is told to write its outputs under
# _SANDBOX_OUTPUTS_DIR so we can download them back afterwards.
_SANDBOX_INPUTS_DIR = "/tmp/diffusers-cli/inputs"
_SANDBOX_OUTPUTS_DIR = "/tmp/diffusers-cli/outputs"

RUN_ID_ENV = "DIFFUSERS_CLI_RUN_ID"

# Namespace keys that control *how* a remote run is dispatched, not what the sandbox CLI
# runs. They are stripped when forwarding argv to the sandbox.
REMOTE_KEYS = frozenset(
    {
        "remote",
        "flavor",
        "timeout",
        "dependencies",
        "namespace",
        "image",
        "keep_alive",
        "sandbox_id",
        "idle_timeout",
        "volume",
        "func",
        "format",  # top-level --format is a local rendering flag; never forward to the sandbox
    }
)


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------


def _add_loading_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--model", "-m", required=True, help="Model id on the Hugging Face Hub or local path.")
    parser.add_argument(
        "--device-map",
        default=None,
        help=(
            "Component placement. Accepts a torch device string (`cuda`, `cuda:0`, `cpu`, `mps`), "
            "`balanced` for pipeline-level auto-split across visible GPUs, or a JSON dict of "
            '`{"<component>": <device>}` for explicit per-component placement. Auto-detected if omitted.'
        ),
    )
    parser.add_argument("--dtype", default="auto", choices=DTYPE_CHOICES, help="Torch dtype for pipeline weights.")
    parser.add_argument("--variant", default=None, help='Optional weight variant (e.g. "fp16").')
    parser.add_argument("--revision", default=None, help="Model revision (branch, tag, or commit SHA).")
    parser.add_argument("--token", default=None, help="Hugging Face token for gated/private models.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom code from the Hub.")
    parser.add_argument(
        "--lora",
        action="append",
        default=None,
        metavar="JSON",
        help=(
            "JSON dict describing a LoRA adapter to attach after the pipeline loads. Repeat to stack "
            'multiple adapters. Format: \'{"lora_id": "<id>", "lora_scale": <float>}\'. `lora_scale` '
            "defaults to 1.0; `adapter_name` is optional (auto-generated as `lora_<i>` when stacking); "
            "`weight_name` picks the weight file when the repo ships more than one."
        ),
    )


def _add_optimization_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--cpu-offload",
        choices=CPU_OFFLOAD_CHOICES,
        default=None,
        help=(
            "Offload pipeline components to CPU during inference. "
            "'model' uses enable_model_cpu_offload, "
            "'group' uses pipeline.enable_group_offload(leaf_level, use_stream=True). "
            "Modular pipelines only support 'auto', which offloads through a ComponentsManager "
            "via enable_auto_cpu_offload."
        ),
    )
    parser.add_argument(
        "--attention-backend",
        choices=ATTENTION_BACKEND_CHOICES,
        default="default",
        help=(
            "Override the attention backend on the transformer/UNet. "
            "Only Hub-hosted kernels are exposed — they auto-download on first use."
        ),
    )
    parser.add_argument("--vae-tiling", action="store_true", help="Enable VAE tiling (lower peak VRAM).")
    parser.add_argument("--vae-slicing", action="store_true", help="Enable VAE slicing (lower peak VRAM).")
    parser.add_argument(
        "--context-parallel",
        action="store_true",
        help=(
            "Enable Ulysses-style context parallelism (ulysses_anything mode). "
            "Requires a DiT-based pipeline and launching the CLI under torchrun with ≥2 GPUs."
        ),
    )
    parser.add_argument(
        "--compile",
        nargs="?",
        const='{"fullgraph": true}',
        default=None,
        metavar="JSON",
        help=(
            "torch.compile every denoiser submodule on the pipeline. Accepts an optional JSON "
            'object of kwargs forwarded to `torch.compile`, e.g. \'{"mode": "max-autotune", '
            '"fullgraph": true}\'. Bare `--compile` uses `fullgraph=true`. Adds a one-time '
            "compilation cost on the first step but speeds up every subsequent step — worth it "
            "for multi-step generation (50+ steps)."
        ),
    )


def _add_output_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Output file or directory. Defaults to "
            "~/.diffusers/cli/run/outputs/diffusers-run-<YYYYMMDDTHHMMSS>-<short-uuid>/<NNNN>.<ext>."
        ),
    )
    parser.add_argument(
        "--push-to",
        default=None,
        help=(
            "Upload the generated files to this HF bucket after saving (created if missing). Accepts "
            "an HF bucket id (`<namespace>/<name>`), an `hf://buckets/<namespace>/<name>[/<subpath>]` "
            "URI, or a browser URL for the same — a subpath is used as a folder prefix. Under --remote "
            "the upload runs inside the sandbox; without an explicit --output the bucket becomes the "
            "sole destination and nothing is downloaded back."
        ),
    )


def _add_remote_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Run this command in a Hugging Face Sandbox instead of on the local machine.",
    )
    parser.add_argument(
        "--flavor",
        default="a10g-small",
        help="HF Sandbox hardware flavor for --remote (e.g. a10g-small, a100-large, cpu-basic).",
    )
    parser.add_argument(
        "--timeout",
        default="10m",
        help="Max wallclock for the run command inside the sandbox (e.g. 30m, 2h). Defaults to 10m.",
    )
    parser.add_argument(
        "--dependencies",
        action="append",
        default=None,
        help="Extra pip dependencies to install in the sandbox. Repeat to add multiple.",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="HF namespace to create the sandbox under (defaults to the current user).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help=(
            "Sandbox image for --remote (defaults to "
            f"{_DEFAULT_REMOTE_IMAGE!r}). Must provide torch + CUDA; the CLI installs the "
            "small Python deps on top via `uv pip install --system`."
        ),
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help=(
            "Don't terminate the sandbox after the run. Its id is printed so a later --remote run "
            "can reconnect with --sandbox-id and reuse the warm deps/weights/compile cache."
        ),
    )
    parser.add_argument(
        "--sandbox-id",
        default=None,
        help=(
            "Reconnect to an existing sandbox (from a prior --keep-alive run) instead of creating a new "
            "one, reusing its warm deps/weights/compile cache. Implies --keep-alive; stop it with "
            "`hf sandbox kill <id>`."
        ),
    )
    parser.add_argument(
        "--idle-timeout",
        default="10m",
        help=(
            "Auto-shutdown the sandbox after this much inactivity (e.g. 30m, 1h). Defaults to 10m. "
            "Only applied on new sandbox creation — ignored when reconnecting via --sandbox-id."
        ),
    )
    parser.add_argument(
        "--volume",
        action="append",
        default=None,
        metavar="BUCKET_ID[:MOUNT_PATH]",
        help=(
            "Mount an HF bucket into the sandbox as a read-write directory. Repeatable. Format: "
            "`<namespace>/<name>` (mounts at `/mnt/buckets/<namespace>/<name>`) or "
            "`<namespace>/<name>:/some/path` for a custom path. Reference mounted files from "
            "--pipeline-kwargs like any other local path. Applied only on new sandbox creation — "
            "ignored when reconnecting via --sandbox-id."
        ),
    )


# ---------------------------------------------------------------------------
# Pipeline loading + optimization
# ---------------------------------------------------------------------------


def _resolve_dtype(name: str | None):
    if name in (None, "auto"):
        return "auto"

    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unknown dtype: {name}")
    return mapping[name]


def _resolve_device_map(raw: str | None) -> str | dict:
    """Parse `--device-map` into a value acceptable by `from_pretrained(device_map=...)`.

    Returns a JSON dict if the value looks like one, `"balanced"` verbatim, or a single-device string (e.g. `"cuda"`,
    `"cuda:1"`, `"cpu"`, `"mps"`). Auto-detects when `raw is None`, pinning to `cuda:$LOCAL_RANK` under torchrun.
    """
    if raw is None:
        if torch_device == "cuda":
            local_rank = os.environ.get("LOCAL_RANK")
            if local_rank is not None:
                torch.cuda.set_device(int(local_rank))
                return f"cuda:{local_rank}"
        return torch_device

    if raw.strip().startswith("{"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--device-map must be a device string or a JSON dict: {e}") from e
        if not isinstance(parsed, dict):
            raise SystemExit("--device-map JSON must decode to an object.")
        return parsed

    return raw


def _apply_cpu_offload(pipeline: Any, mode: str, device_map: str | dict, offload_margin: str | None = None) -> None:
    """Apply CPU offload. Requires a single-device target (not balanced or dict).

    Standard pipelines support 'model' and 'group'; modular pipelines offload through the ComponentsManager they were
    loaded with ('auto').
    """
    if not isinstance(device_map, str) or device_map == "balanced":
        raise SystemExit(
            "--cpu-offload requires --device-map to be a single device string (e.g. 'cuda'); "
            f"got {device_map!r}. balanced/dict placement is incompatible with CPU offload."
        )

    if isinstance(pipeline, diffusers.ModularPipeline):
        offload_kwargs = {"memory_reserve_margin": offload_margin} if offload_margin is not None else {}
        pipeline._components_manager.enable_auto_cpu_offload(device=device_map, **offload_kwargs)
        return

    if mode == "auto":
        raise SystemExit(
            "--cpu-offload auto only applies to modular pipelines (it offloads through a "
            "ComponentsManager). Use 'model' or 'group' for standard pipelines."
        )
    if offload_margin is not None:
        logger.warning(
            f"--offload-margin {offload_margin!r} only applies to `--cpu-offload auto`; "
            f"ignoring it for `--cpu-offload {mode}`."
        )
    if mode == "model":
        pipeline.enable_model_cpu_offload(device=device_map)
    elif mode == "group":
        pipeline.enable_group_offload(
            onload_device=torch.device(device_map),
            offload_type="leaf_level",
            use_stream=True,
        )


def _set_attention_backend(pipeline: Any, backend: str) -> None:
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None or not hasattr(transformer, "set_attention_backend"):
        logger.warning(
            f"--attention-backend is only supported on transformer-based pipelines; "
            f"{type(pipeline).__name__} uses the legacy UNet attention path."
        )
        return
    try:
        transformer.set_attention_backend(backend)
    except (ValueError, ImportError, RuntimeError) as e:
        logger.warning(
            f"Attention backend {backend!r} could not be set on {type(transformer).__name__}: "
            f"{type(e).__name__}: {e}. Falling back to the model's default backend."
        )


def _enable_context_parallel(pipeline: Any) -> None:
    if not torch.distributed.is_available():
        raise SystemExit("--context-parallel requires a torch build with distributed support.")

    if not torch.distributed.is_initialized():
        # Hybrid backend: ulysses_anything's per-rank size coordination wants Gloo on CPU
        # (avoids H2D/D2H for a tiny int tensor); the main attention all-to-all stays on NCCL.
        torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl")

    transformer = getattr(pipeline, "transformer", None)
    if transformer is None or not hasattr(transformer, "enable_parallelism"):
        raise SystemExit(
            "--context-parallel requires a DiT-based pipeline. "
            f"{type(pipeline).__name__} does not expose a `transformer` with `enable_parallelism`."
        )

    transformer.enable_parallelism(
        config=ContextParallelConfig(
            ulysses_degree=torch.distributed.get_world_size(),
            ring_degree=1,
            ulysses_anything=True,
        )
    )


def _apply_optimizations(pipeline: Any, args: Namespace) -> None:
    """Apply VAE tiling/slicing, attention backend, context-parallel, and torch.compile toggles."""
    vae = getattr(pipeline, "vae", None)
    if args.vae_tiling and vae is not None and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if args.vae_slicing and vae is not None and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if args.attention_backend != "default":
        _set_attention_backend(pipeline, args.attention_backend)
    if args.context_parallel:
        _enable_context_parallel(pipeline)
    if args.compile is not None:
        if args.context_parallel:
            logger.warning("--compile is currently not supported with --context-parallel; skipping compile.")
        else:
            _compile_denoiser(pipeline, args.compile)


def _compile_denoiser(pipeline: Any, compile_spec: str) -> None:
    """Compile every `transformer*` and `unet*` submodule on the pipeline.

    `compile_spec` is the raw JSON string from `--compile` (`"{}"` for bare flag). Decoded into kwargs and forwarded
    verbatim to the compile call.

    Prefers regional compilation via `module.compile_repeated_blocks(**kwargs)` — only compiles the repeated inner
    blocks (the bulk of the compute), much faster first-step latency than compiling the whole module. Falls back to
    full `torch.compile` if the model doesn't expose `_repeated_blocks`.
    """

    try:
        compile_kwargs = json.loads(compile_spec)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--compile must be valid JSON: {e}") from e
    if not isinstance(compile_kwargs, dict):
        raise SystemExit("--compile must decode to a JSON object.")

    for attr in dir(pipeline):
        if not any(attr.startswith(key) for key in _DENOISER_COMPONENT_KEYS):
            continue
        module = getattr(pipeline, attr, None)
        if not isinstance(module, torch.nn.Module):
            continue

        if getattr(module, "_repeated_blocks", None):
            # Regional compile — only the repeated blocks. Mutates `module` in place.
            module.compile_repeated_blocks(**compile_kwargs)
        else:
            # No regional metadata declared; fall back to compiling the whole module.
            setattr(pipeline, attr, torch.compile(module, **compile_kwargs))


def _load_lora(pipeline: Any, args: Namespace) -> None:
    """Attach one or more LoRA adapters. Each `--lora` value is a JSON dict.

    Per-entry fields: `lora_id` (required), `lora_scale` (optional float, default 1.0), `adapter_name` (optional;
    auto-generated as `lora_<i>` when stacking), `weight_name` (optional; required when the repo ships more than one
    weight file, e.g. a ComfyUI variant alongside the diffusers one). Multiple `--lora` flags stack via a single
    `set_adapters(...)` call at the end.
    """
    if not args.lora:
        return
    specs = []
    for raw in args.lora:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--lora must be valid JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise SystemExit(f"--lora must decode to a JSON object; got {type(parsed).__name__}.")
        specs.append(parsed)
    if not hasattr(pipeline, "load_lora_weights"):
        raise SystemExit(f"{type(pipeline).__name__} does not support LoRA loading.")

    names: list[str] = []
    scales: list[float] = []
    for i, spec in enumerate(specs):
        lora_id = spec.get("lora_id")
        if not lora_id:
            raise SystemExit(f"--lora entry {i} is missing 'lora_id'.")
        adapter_name = spec.get("adapter_name") or (f"lora_{i}" if len(specs) > 1 else "default")
        pipeline.load_lora_weights(lora_id, adapter_name=adapter_name, weight_name=spec.get("weight_name", None))
        names.append(adapter_name)
        scales.append(float(spec.get("lora_scale", 1.0)))

    if hasattr(pipeline, "set_adapters"):
        pipeline.set_adapters(names, adapter_weights=scales)


def _load_pipeline(args: Namespace) -> Any:
    # Detect modular repos by trying the standard config; `ModularPipeline` repos ship
    # `modular_model_index.json` instead of `model_index.json`, so `load_config` OSErrors.
    # A repo can also ship a `model_index.json` whose `_class_name` is a modular pipeline
    # (e.g. MiniMax-H3's repository-level entry), so route on the class, not just the file.
    try:
        config = diffusers.DiffusionPipeline.load_config(args.model, token=args.token, revision=args.revision)
        cls = getattr(diffusers, str(config.get("_class_name")), None)
        modular = isinstance(cls, type) and issubclass(cls, diffusers.ModularPipeline)
    except OSError:
        modular = True

    dtype = _resolve_dtype(args.dtype)
    device_map = _resolve_device_map(args.device_map)
    common_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    if dtype != "auto":
        common_kwargs["torch_dtype"] = dtype
    if args.variant:
        common_kwargs["variant"] = args.variant
    if args.token:
        common_kwargs["token"] = args.token
    # CPU offload sets up its own placement hooks, so leave weights on CPU at load time.
    if not args.cpu_offload:
        common_kwargs["device_map"] = device_map

    if modular:
        if args.cpu_offload and args.cpu_offload != "auto":
            raise SystemExit(
                f"--cpu-offload {args.cpu_offload!r} is not supported for modular pipelines — they "
                "offload through a ComponentsManager. Use `--cpu-offload auto`."
            )
        components_manager = diffusers.ComponentsManager() if args.cpu_offload else None
        # ModularPipeline.from_pretrained fetches only the pipeline config; component
        # weights come in via load_components(). `revision` scopes the config fetch,
        # so it stays on from_pretrained — each ComponentSpec pins its own revision,
        # and forwarding a global `revision` to load_components() would override those.
        pipeline = diffusers.ModularPipeline.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            token=args.token,
            revision=args.revision,
            components_manager=components_manager,
            workflow=args.workflow,
        )
        pipeline.load_components(**common_kwargs)
    else:
        if args.workflow:
            logger.warning(
                f"--workflow {args.workflow!r} only applies to modular pipelines; "
                f"ignoring it — '{args.model}' loads as a standard DiffusionPipeline."
            )
        pipeline = diffusers.DiffusionPipeline.from_pretrained(args.model, revision=args.revision, **common_kwargs)

    _load_lora(pipeline, args)
    if args.cpu_offload:
        _apply_cpu_offload(pipeline, args.cpu_offload, device_map, args.offload_margin)
    _apply_optimizations(pipeline, args)

    return pipeline


# ---------------------------------------------------------------------------
# Pipeline call helpers
# ---------------------------------------------------------------------------


def _parse_pipeline_kwargs(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--pipeline-kwargs must be valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise SystemExit("--pipeline-kwargs must decode to a JSON object.")
    return parsed


def _load_audio(url_or_path: str) -> tuple[Any, int]:
    """Load audio from a URL or local path via torchaudio. Returns `(waveform, sampling_rate)`."""
    import torchaudio

    if url_or_path.startswith(("http://", "https://")):
        resp = httpx.get(url_or_path, follow_redirects=True, timeout=DIFFUSERS_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return torchaudio.load(io.BytesIO(resp.content))
    return torchaudio.load(url_or_path)


def _resolve_media_inputs(call_kwargs: dict[str, Any]) -> None:
    """Replace string paths/URLs at known media-input keys with loaded tensors.

    Images resolve to `PIL.Image.Image` via `load_image`; videos to `list[PIL.Image.Image]` via `load_video`; audio to
    a `torch.Tensor` via `_load_audio` (also auto-sets the paired sampling-rate kwarg for `initial_audio_waveforms` if
    the user didn't supply it). A `list[str]` at any key is treated as a batch: each entry is loaded and the value
    becomes a list of loaded objects. Non-string, non-list values pass through untouched.
    """

    def _is_string_list(v: Any) -> bool:
        return isinstance(v, list) and bool(v) and all(isinstance(x, str) for x in v)

    for key in _IMAGE_INPUT_KEYS:
        value = call_kwargs.get(key)
        if isinstance(value, str):
            call_kwargs[key] = load_image(value)
        elif _is_string_list(value):
            call_kwargs[key] = [load_image(v) for v in value]
    for key in _VIDEO_INPUT_KEYS:
        value = call_kwargs.get(key)
        if isinstance(value, str):
            call_kwargs[key] = load_video(value)
        elif _is_string_list(value):
            call_kwargs[key] = [load_video(v) for v in value]
    for key in _AUDIO_INPUT_KEYS:
        value = call_kwargs.get(key)
        if isinstance(value, str):
            waveform, sr = _load_audio(value)
            call_kwargs[key] = waveform
            if key == "initial_audio_waveforms" and "initial_audio_sampling_rate" not in call_kwargs:
                call_kwargs["initial_audio_sampling_rate"] = sr
        elif _is_string_list(value):
            pairs = [_load_audio(v) for v in value]
            call_kwargs[key] = [w for w, _ in pairs]
            if key == "initial_audio_waveforms" and "initial_audio_sampling_rate" not in call_kwargs:
                # All batched waveforms must share a sampling rate; use the first entry's.
                call_kwargs["initial_audio_sampling_rate"] = pairs[0][1]


def _get_generator(seed: int | None, device: str):
    if seed is None:
        return None

    generator_device = "cpu" if device == "mps" else device
    return torch.Generator(device=generator_device).manual_seed(seed)


def _unwrap_pipeline_output(result: Any) -> list[Any]:
    """Resolve a pipeline-output object into the media payloads the saver dispatches on.

    An output can carry more than one media field (e.g. LTX2 returns video in `frames` and a waveform in `audio`), so
    every known field that is present is saved, not just the first match. Payloads keep their batch dimension —
    `_save_output` dispatches on the full batched shape.
    """
    payloads = [
        getattr(result, name)
        for name in ("images", "frames", "audios", "audio")
        if getattr(result, name, None) is not None
    ]
    return payloads or [result]


# ---------------------------------------------------------------------------
# Output saving (dispatch by type)
# ---------------------------------------------------------------------------


def _get_or_create_run_id() -> str:
    """Return the current run's id, creating one if not yet set.

    Format: `diffusers-run-<YYYYMMDDTHHMMSS>-<6-char-uuid>`. Same id is reused as the local output subdirectory, the
    remote bucket prefix, and the container-side `RUN_ID_ENV` so a run's artifacts are traceable end-to-end.
    """

    existing = os.environ.get(RUN_ID_ENV)
    if existing:
        return existing
    run_id = f"diffusers-run-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    os.environ[RUN_ID_ENV] = run_id
    return run_id


def _resolve_output_paths(num: int, explicit: str | None, ext: str) -> list[Path]:
    if explicit is None:
        base = Path(DEFAULT_OUTPUT_DIR) / _get_or_create_run_id()
        base.mkdir(parents=True, exist_ok=True)
        return [base / f"{i:04d}.{ext}" for i in range(num)]

    p = Path(explicit)
    if explicit.endswith(os.sep) or p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
        return [p / f"{i:04d}.{ext}" for i in range(num)]

    p.parent.mkdir(parents=True, exist_ok=True)
    if num == 1:
        return [p]
    stem, suffix = p.stem, p.suffix or f".{ext}"
    return [p.with_name(f"{stem}-{i:04d}{suffix}") for i in range(num)]


def _as_pil_list(value: Any):
    if isinstance(value, Image.Image):
        return [value]
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, Image.Image) for v in value):
        return list(value)
    return None


def _as_frame_sequence(value: Any):
    if isinstance(value, (list, tuple)) and len(value) >= 2 and isinstance(value[0], (Image.Image, np.ndarray)):
        return list(value)
    return None


def _as_audio_arrays(value: Any):
    if isinstance(value, np.ndarray) and value.ndim <= 2:
        return [value]
    if isinstance(value, np.ndarray) and value.ndim == 3:
        return list(value)
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, np.ndarray) for v in value):
        return list(value)
    return None


def _save_audio_arrays(audios, sampling_rate: int, args: Namespace) -> list[str]:
    """Write each numpy audio array to a 16-bit PCM WAV at `sampling_rate` Hz.

    Uses the stdlib `wave` module so no scipy dependency is required.
    """

    paths = _resolve_output_paths(len(audios), args.output, ext="wav")
    saved: list[str] = []
    for audio, path in zip(audios, paths):
        data = np.asarray(audio)
        if data.dtype.kind == "f":
            data = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            data = data.astype(np.int16)
        if data.ndim == 1:
            n_channels = 1
        else:
            # Heuristic: shorter axis is channels (interleaved layout for `wave` is
            # samples × channels, so transpose if needed).
            if data.shape[0] < data.shape[-1]:
                data = data.T
            n_channels = data.shape[1]
        with wave.open(str(path), "wb") as w:
            w.setnchannels(n_channels)
            w.setsampwidth(2)  # 16-bit PCM
            w.setframerate(sampling_rate)
            w.writeframes(data.tobytes())
        saved.append(str(path))
    return saved


def _save_videos(videos: list[Any], args: Namespace) -> list[str]:
    """Write each frame sequence to mp4, plus every frame as `<video-stem>-frames/<NNNN>.png` beside it.

    The per-video subfolder ties each frame to its video and keeps paths unique across a batch. Frames are written
    first and need no video backend, so they double as the safety net: if `export_to_video` fails (e.g. `imageio`
    missing), the frames are already on disk and only the mp4 is skipped.
    """
    mp4_paths = _resolve_output_paths(len(videos), args.output, ext="mp4")
    saved: list[str] = []
    for frames, path in zip(videos, mp4_paths):
        frames = list(frames)
        frames_dir = path.with_name(f"{path.stem}-frames")
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            if not isinstance(frame, Image.Image):
                arr = np.asarray(frame)
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
                frame = Image.fromarray(arr)
            frame_path = frames_dir / f"{i:04d}.png"
            frame.save(frame_path)
            saved.append(str(frame_path))
        try:
            export_to_video(frames, str(path), fps=args.fps)
            saved.append(str(path))
        except Exception as e:
            logger.warning(
                f"Video export failed ({e}); the individual frames of {path.stem} are saved next to it as PNGs. "
                "Install a video backend with: pip install imageio imageio-ffmpeg"
            )
    return saved


def _save_output(value: Any, args: Namespace) -> list[str]:
    """Save `value` by dispatching on its runtime type and, for arrays, its shape."""
    # Tensors arrive only when the user explicitly asked for `output_type="pt"` (or the pipeline
    # natively defaults to it, e.g. StableAudio). Postprocessed pt outputs are channels-first per
    # frame — (B, C, H, W) images, (B, F, C, H, W) video from `postprocess_video` — while the array
    # branches below expect channels-last, so convert here.
    if isinstance(value, torch.Tensor):
        arr = value.detach().to(torch.float32).cpu().numpy()
        if arr.ndim == 5:
            arr = arr.transpose(0, 1, 3, 4, 2)
        elif arr.ndim == 4:
            arr = arr.transpose(0, 2, 3, 1)
        value = arr

    # Array shapes are unambiguous where PIL lists are not: (B, F, H, W, C) is batched video,
    # (B, H, W, C) is batched images.
    if isinstance(value, np.ndarray):
        if value.ndim == 5:
            return _save_videos(list(value), args)
        if value.ndim == 4:
            paths = _resolve_output_paths(len(value), args.output, ext="png")
            for arr, path in zip(value, paths):
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
                Image.fromarray(arr).save(path)
            return [str(p) for p in paths]

    pil_images = _as_pil_list(value)
    if pil_images is not None:
        paths = _resolve_output_paths(len(pil_images), args.output, ext="png")
        for img, path in zip(pil_images, paths):
            img.save(path)
        return [str(p) for p in paths]

    frames = _as_frame_sequence(value)
    if frames is not None:
        return _save_videos([frames], args)

    # A batch of PIL frame sequences — what video pipelines return for an explicit
    # `output_type="pil"`. Previously this matched no branch and fell through to the JSON dump.
    if isinstance(value, (list, tuple)) and value and all(_as_frame_sequence(v) is not None for v in value):
        return _save_videos([list(v) for v in value], args)

    audios = _as_audio_arrays(value)
    if audios is not None:
        return _save_audio_arrays(audios, args.sampling_rate or 16000, args)

    raise ValueError(
        f"Cannot save pipeline output of type {type(value).__name__!r}: not a recognized image, video, or audio "
        "payload. For modular pipelines, name the media outputs to save with `--output-key` (repeat it for "
        "several, e.g. `--output-key videos --output-key audio`); non-media values like a sample rate are not "
        "saved and are set with their own flag (`--sampling-rate`)."
    )


# ---------------------------------------------------------------------------
# Hub bucket upload (--push-to)
# ---------------------------------------------------------------------------


def _parse_push_to(spec: str) -> tuple[str, str]:
    """Split `--push-to` into a bucket id and an optional subpath prefix.

    Accepts an HF bucket id (`<namespace>/<name>[/<subpath>]`), a canonical
    `hf://buckets/<namespace>/<name>[/<subpath>]` URI, or a Hub web URL for the same. Non-bucket URIs (models,
    datasets, spaces) are rejected — `--push-to` targets storage buckets only.
    """

    # Bare shorthand → canonical URI so a single parser handles every accepted form.
    if not spec.startswith(("hf://", "http://", "https://")):
        spec = f"hf://buckets/{spec.strip('/')}"
    uri = parse_hf_uri(spec)
    if not uri.is_bucket:
        raise SystemExit(f"--push-to must point at a bucket; got {uri.type!r} URI {spec!r}.")
    return uri.id, uri.path_in_repo


def _collapse_frame_dirs(paths: list[str]) -> list[str]:
    """Replace each `<stem>-frames/` group with its directory path for reporting.

    A video's frames are hundreds of files; listing them all buries the mp4 and floods the terminal scrollback. Entries
    stay real paths, so the reported list is still usable programmatically. The files themselves are untouched — only
    what is reported changes.
    """
    collapsed: list[str] = []
    for path in paths:
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        if not parent.endswith("-frames"):
            collapsed.append(path)
        elif parent not in collapsed:
            collapsed.append(parent)
    return collapsed


def _push_outputs(args: Namespace, saved_paths: list[str]) -> dict[str, Any] | None:
    """Upload `saved_paths` to the `--push-to` bucket. Returns a summary or None."""
    if not args.push_to:
        return None

    bucket_id, subpath = _parse_push_to(args.push_to)
    api = HfApi(token=args.token)
    api.create_bucket(bucket_id, exist_ok=True)

    run_id = _get_or_create_run_id()
    prefix = f"{subpath}/{run_id}" if subpath else run_id
    # Destination paths keep their structure relative to the common output dir, so files in
    # subfolders (e.g. `0000-frames/0003.png`) don't collide on basename in the bucket.
    base = Path(os.path.commonpath([str(Path(p).parent) for p in saved_paths]))
    add = [(local, f"{prefix}/{Path(local).relative_to(base).as_posix()}") for local in saved_paths]
    api.batch_bucket_files(bucket_id, add=add)

    uploaded = _collapse_frame_dirs([f"hf://buckets/{bucket_id}/{dest}" for _, dest in add])
    return {"bucket_id": bucket_id, "uploaded": uploaded}


# ---------------------------------------------------------------------------
# Remote execution (HF Sandbox)
# ---------------------------------------------------------------------------


def _build_task_kwargs(args: Namespace) -> dict[str, Any]:
    """Pick out the kwargs the sandbox CLI should invoke the task with."""
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in REMOTE_KEYS or value is None or value is False:
            continue
        out[key] = value
    return out


def _kwargs_to_argv(task: str, task_kwargs: dict[str, Any]) -> list[str]:
    """Render `task_kwargs` as the argv list the sandbox CLI's argparse will see."""
    argv: list[str] = [task]
    for key, value in task_kwargs.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif isinstance(value, list):
            for item in value:
                argv.extend([flag, str(item)])
        else:
            argv.extend([flag, str(value)])
    return argv


def _duration_to_seconds(value: str) -> float:
    """Parse a duration like `30s`, `10m`, `2h` (or a bare number of seconds) into seconds."""
    value = value.strip()
    units = {"s": 1, "m": 60, "h": 3600}
    if value and value[-1] in units:
        return float(value[:-1]) * units[value[-1]]
    return float(value)


def _upload_inputs_to_sandbox(args: Namespace, sbx: Any, run_id: str) -> None:
    """Upload local media paths in `--pipeline-kwargs` into the sandbox and rewrite the JSON in place.

    Walks known image/video/audio-input keys; any string value that resolves to a local file is uploaded to
    `<_SANDBOX_INPUTS_DIR>/<run_id>/<key>_<basename>` and the JSON path is rewritten to that in-sandbox path. URLs,
    `hf://` URIs, and non-existent paths pass through untouched.
    """
    if not args.pipeline_kwargs:
        return
    try:
        parsed = json.loads(args.pipeline_kwargs)
    except json.JSONDecodeError:
        return  # the sandbox CLI will fail loudly with a parse error later
    if not isinstance(parsed, dict):
        return

    def _upload_one(key: str, index: int | None, local_str: str) -> str:
        # `index` is None for scalar entries, an int for list entries (used to disambiguate names).
        local = Path(local_str)
        suffix = f"_{index}" if index is not None else ""
        remote_path = f"{_SANDBOX_INPUTS_DIR}/{run_id}/{key}{suffix}_{local.name}"
        sbx.files.upload(str(local), remote_path)
        return remote_path

    uploaded = 0
    for key in (*_IMAGE_INPUT_KEYS, *_VIDEO_INPUT_KEYS, *_AUDIO_INPUT_KEYS):
        value = parsed.get(key)
        if isinstance(value, str) and Path(value).is_file():
            parsed[key] = _upload_one(key, None, value)
            uploaded += 1
        elif isinstance(value, list):
            # Batched inputs: upload each local path, leave URLs/hf:// URIs alone.
            new_list = list(value)
            for i, entry in enumerate(value):
                if isinstance(entry, str) and Path(entry).is_file():
                    new_list[i] = _upload_one(key, i, entry)
                    uploaded += 1
            parsed[key] = new_list

    if uploaded:
        logger.info(f"uploaded {uploaded} local input file(s) to the sandbox")
        args.pipeline_kwargs = json.dumps(parsed)


def _download_outputs_from_sandbox(sbx: Any, sandbox_dir: str, local_dir: Path) -> list[str]:
    """Download everything the sandbox CLI wrote under `sandbox_dir` into `local_dir`, recursing into subfolders."""
    local_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for entry in sbx.files.list(sandbox_dir):
        if entry.type == "dir":
            saved.extend(_download_outputs_from_sandbox(sbx, entry.path, local_dir / Path(entry.path).name))
        elif entry.type == "file":
            target = local_dir / Path(entry.path).name
            sbx.files.download(entry.path, str(target))
            saved.append(str(target))
    return saved


def _maybe_submit_remote(args: Namespace, task: str) -> bool:
    """If `--remote` was set, run this invocation inside an HF Sandbox and return True."""
    if not args.remote:
        return False

    if Path(args.model).exists():
        raise SystemExit(
            f"--model {args.model!r} is a local path; the sandbox can't see it. "
            "Pass a Hub repo id so the sandbox can download it."
        )

    hf_token = args.token or get_token()
    run_id = _get_or_create_run_id()

    # An explicit --push-to means the bucket is the user's destination, so skip the local
    # download unless they also asked for a local path via --output.
    user_bucket = bool(args.push_to)
    download_locally = (not user_bucket) or (args.output is not None)
    local_dir = Path(args.output) if args.output else Path(DEFAULT_OUTPUT_DIR) / run_id

    use_existing_sandbox = bool(args.sandbox_id)
    keep_alive = args.keep_alive or use_existing_sandbox
    if use_existing_sandbox and args.volume:
        logger.warning(
            "--volume is ignored when reconnecting to an existing sandbox (mounts are set at creation time)."
        )
    if use_existing_sandbox:
        logger.info(f"reconnecting to sandbox {args.sandbox_id!r}...")
        sbx = Sandbox.connect(args.sandbox_id, token=hf_token)
    else:
        logger.info(f"creating sandbox on flavor={args.flavor!r}...")
        create_kwargs: dict[str, Any] = {
            "image": args.image or _DEFAULT_REMOTE_IMAGE,
            "flavor": args.flavor,
            "forward_hf_token": True,
            "token": hf_token,
            "env": {
                "HF_ENABLE_PARALLEL_LOADING": "1",
                "DIFFUSERS_VERBOSITY": os.environ.get("DIFFUSERS_VERBOSITY", "info"),
            },
            "idle_timeout": args.idle_timeout,
            # The 120s default expires while a cold node pulls the multi-GB pytorch base image or
            # waits for GPU capacity, which fails the run before it starts.
            "start_timeout": 600.0,
        }
        if args.volume:
            volumes = []
            for spec in args.volume:
                bucket_id, sep, mount_path = spec.partition(":")
                if not sep:
                    mount_path = f"/mnt/buckets/{bucket_id}"
                if bucket_id.count("/") != 1:
                    raise SystemExit(f"--volume: bucket id must be <namespace>/<name>, got {bucket_id!r}")
                if not mount_path.startswith("/"):
                    raise SystemExit(f"--volume: mount path must be absolute, got {mount_path!r}")
                volumes.append(Volume(type="bucket", source=bucket_id, mount_path=mount_path))
            create_kwargs["volumes"] = volumes
        if args.namespace is not None:
            create_kwargs["namespace"] = args.namespace
        sbx = Sandbox.create(**create_kwargs)

    def _stream(chunk: str) -> None:
        sys.stderr.write(chunk)
        sys.stderr.flush()

    exit_code = 0
    saved: list[str] = []
    run_seconds = 0.0
    try:
        _upload_inputs_to_sandbox(args, sbx, run_id)

        dependencies = list(_DEFAULT_REMOTE_DEPS)
        if args.dependencies:
            dependencies.extend(args.dependencies)
        # --break-system-packages bypasses PEP 668; harmless in a throwaway sandbox. uv is a
        # near no-op when the deps are already satisfied, so this stays cheap on a reused sandbox.
        install_cmd = shlex.join(["uv", "pip", "install", "--system", "--break-system-packages", *dependencies])
        logger.info("installing dependencies in the sandbox...")
        sbx.run(install_cmd, on_stdout=_stream, on_stderr=_stream)

        # Per-run outputs subdirectory so a reused sandbox doesn't leak files from prior runs
        # into this run's download set.
        sandbox_output_dir = f"{_SANDBOX_OUTPUTS_DIR}/{run_id}"
        task_kwargs = _build_task_kwargs(args)
        task_kwargs["output"] = sandbox_output_dir + "/"
        cli_argv = _kwargs_to_argv(task, task_kwargs)
        # Suppress the container CLI's own `out.result(...)` payload — the outer wrapper owns the
        # final structured output for --remote runs.
        format_argv = ["--format", "quiet"]
        # torchrun wraps the CLI for --context-parallel so torch.distributed initializes across
        # every visible GPU before the run command starts.
        if args.context_parallel:
            cli_argv = [
                "torchrun",
                "--nproc-per-node=gpu",
                "-m",
                "diffusers.commands.diffusers_cli",
                *format_argv,
                *cli_argv,
            ]
        else:
            cli_argv = [_CONTAINER_CLI_BINARY, *format_argv, *cli_argv]

        started = time.perf_counter()
        # Per-invocation env: RUN_ID_ENV must be fresh each run. Sandbox.create-time env is
        # baked in and would go stale on reused sandboxes, silently reusing the initial run's
        # bucket prefix in `_push_outputs`.
        result = sbx.run(
            cli_argv,
            env={RUN_ID_ENV: run_id},
            on_stdout=_stream,
            on_stderr=_stream,
            timeout=_duration_to_seconds(args.timeout),
            check=False,
        )
        run_seconds = time.perf_counter() - started
        exit_code = result.exit_code

        if exit_code == 0 and download_locally:
            saved = _download_outputs_from_sandbox(sbx, sandbox_output_dir, local_dir)
    finally:
        if keep_alive:
            logger.info(
                f"sandbox {sbx.id} kept alive — reconnect with "
                f"`--remote --sandbox-id {sbx.id}`, stop with `hf sandbox kill {sbx.id}`."
            )
        else:
            sbx.kill()

    send_telemetry(
        topic="diffusers/cli/run/remote",
        library_name="diffusers",
        library_version=diffusers.__version__,
    )

    payload: dict[str, Any] = {
        "exit_code": exit_code,
        "run_seconds": round(run_seconds, 1),
    }
    if keep_alive:
        payload["sandbox_id"] = sbx.id
    if download_locally:
        payload["outputs"] = _collapse_frame_dirs(saved)
    if args.push_to:
        bucket_id, subpath = _parse_push_to(args.push_to)
        prefix = f"{subpath}/{run_id}" if subpath else run_id
        payload["pushed-to"] = f"hf://buckets/{bucket_id}/{prefix}/"
    out.result("remote-run", **payload)

    if exit_code != 0:
        raise SystemExit(f"remote run failed with exit code {exit_code}")
    return True


# ---------------------------------------------------------------------------
# Subcommand
# ---------------------------------------------------------------------------


class RunCommand(BaseDiffusersCLICommand):
    task = "run"

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        epilog = (
            "Examples\n"
            "  $ diffusers-cli run -m black-forest-labs/FLUX.1-dev --dtype bf16 \\\n"
            '      --pipeline-kwargs \'{"prompt": "a cat on the moon"}\'\n'
            "  $ diffusers-cli run -m black-forest-labs/FLUX.1-dev --dtype bf16 \\\n"
            '      --pipeline-kwargs \'{"prompt": "make the fur grey", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"}\'\n'
            "  $ diffusers-cli run -m black-forest-labs/FLUX.1-dev --dtype bf16 \\\n"
            '      --pipeline-kwargs \'{"prompt": "a tiny cat"}\' \\\n'
            '      --lora \'{"lora_id": "alvdansen/littletinies", "lora_scale": 0.8}\'\n'
            "  $ diffusers-cli run -m black-forest-labs/FLUX.1-dev --dtype bf16 \\\n"
            '      --pipeline-kwargs \'{"prompt": "a cat"}\' --remote --flavor a100-large\n'
            "  $ diffusers-cli run -m black-forest-labs/FLUX.1-dev --dtype bf16 --context-parallel \\\n"
            '      --pipeline-kwargs \'{"prompt": "a cat"}\' --remote --flavor 4xa100-large\n'
            "\n"
            "Learn more\n"
            "  Use `diffusers-cli <command> --help` for more information about a command.\n"
            "  Read the documentation at https://huggingface.co/docs/diffusers\n"
        )

        parser: ArgumentParser = subparsers.add_parser(
            "run",
            help="Run any diffusers pipeline locally or remotely in an HF Sandbox.",
            usage="\n  diffusers-cli run [options]",
            epilog=epilog,
            formatter_class=RawDescriptionHelpFormatter,
        )
        parser._optionals.title = "Options"
        _add_loading_arguments(parser)
        _add_optimization_arguments(parser)
        parser.add_argument(
            "--pipeline-kwargs",
            default=None,
            help=(
                "JSON object of kwargs passed to the pipeline call. String values at known "
                f"image-input keys ({', '.join(_IMAGE_INPUT_KEYS)}) are auto-loaded as PIL images; "
                f"video-input keys ({', '.join(_VIDEO_INPUT_KEYS)}) are auto-loaded as frame lists; "
                f"audio-input keys ({', '.join(_AUDIO_INPUT_KEYS)}) are auto-loaded via torchaudio."
            ),
        )
        parser.add_argument(
            "--output-key",
            action="append",
            default=None,
            metavar="NAME",
            help=(
                "For modular pipelines: name of the intermediate to extract (passed as `output=` to the call). "
                "Repeat to request several, e.g. `--output-key videos --output-key audio`; each is saved by its "
                "own media type."
            ),
        )
        parser.add_argument(
            "--workflow",
            default=None,
            help=(
                "For modular pipelines: workflow to load (passed to `ModularPipeline.from_pretrained`). "
                "Prunes the blocks to that workflow so `load_components` only fetches its components."
            ),
        )
        parser.add_argument(
            "--offload-margin",
            default=None,
            help=(
                "For `--cpu-offload auto`: device memory kept free for activations "
                "(passed to `ComponentsManager.enable_auto_cpu_offload` as `memory_reserve_margin`, "
                "default 3GB). Raise it when a large canvas OOMs mid-forward."
            ),
        )
        parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
        parser.add_argument(
            "--fps",
            type=int,
            default=8,
            help="FPS used when the output happens to be a frame sequence.",
        )
        parser.add_argument(
            "--sampling-rate",
            type=int,
            default=None,
            help="Sample rate used when the output happens to be an audio array.",
        )
        _add_remote_arguments(parser)
        _add_output_arguments(parser)
        parser.set_defaults(func=RunCommand)

    def __init__(self, args: Namespace):
        self.args = args

    def run(self) -> None:
        _get_or_create_run_id()  # populate RUN_ID_ENV so local output dir + remote bucket prefix agree

        call_kwargs = _parse_pipeline_kwargs(self.args.pipeline_kwargs)

        if _maybe_submit_remote(self.args, self.task):
            return

        # Resolve media before loading pipeline weights so dead URLs / missing files fail
        # fast — cheap to fetch, expensive to load a 20GB model just to hit a 404.
        _resolve_media_inputs(call_kwargs)
        pipeline = _load_pipeline(self.args)
        is_modular = isinstance(pipeline, diffusers.ModularPipeline)

        if self.args.output_key is not None:
            # One key returns that value directly; several return a dict keyed by name.
            keys = self.args.output_key
            call_kwargs["output"] = keys[0] if len(keys) == 1 else keys

        device = pipeline.device.type if hasattr(pipeline, "device") else "cpu"
        generator = _get_generator(self.args.seed, device)
        if generator is not None:
            call_kwargs["generator"] = generator

        try:
            result = pipeline(**call_kwargs)

            # Under torchrun, ranks > 0 produce identical output to rank 0 (CP shards the
            # transformer compute but ranks reduce to the same final tensors). Save/push/print
            # from rank 0 only to avoid clobbering bucket files 4x and printing 4x.
            if os.environ.get("RANK", "0") == "0":
                if not is_modular:
                    savables = _unwrap_pipeline_output(result)
                elif isinstance(result, dict):
                    # Several `--output-key`s: one payload per requested name.
                    savables = list(result.values())
                else:
                    savables = [result]
                saved = []
                for savable in savables:
                    saved.extend(_save_output(savable, self.args))
                pushed = _push_outputs(self.args, saved)

                out.result(
                    self.task,
                    model=self.args.model,
                    device=device,
                    pipeline_class=type(pipeline).__name__,
                    modular=is_modular,
                    outputs=_collapse_frame_dirs(saved),
                    pushed=pushed,
                    seed=self.args.seed,
                    output_key=self.args.output_key,
                )
        finally:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
