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

"""``diffusers-cli generate`` — single agentic entry point.

Runs any diffusers pipeline (standard or modular) by forwarding ``--pipeline-kwargs`` verbatim, saves the output by
sniffing its runtime type, and can submit the same call to HF Jobs via ``--remote``.
"""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser, Namespace, _SubParsersAction
from pathlib import Path
from typing import Any

from diffusers.utils import load_image

from . import BaseDiffusersCLICommand
from ._common import try_fetch_config
from ._output import out


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "outputs"
DTYPE_CHOICES = ("auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32")
CPU_OFFLOAD_CHOICES = ("model", "group")


def _hub_attention_backends() -> tuple[str, ...]:
    """Hub-hosted attention backends sourced from ``_HUB_KERNELS_REGISTRY``.

    Single source of truth: if the registry grows or shrinks, the CLI choices follow.
    """
    from diffusers.models.attention_dispatch import _HUB_KERNELS_REGISTRY

    return tuple(sorted(backend.value for backend in _HUB_KERNELS_REGISTRY))


ATTENTION_BACKEND_CHOICES = ("default", *_hub_attention_backends())

# Keys whose string value should be resolved via ``diffusers.utils.load_image``
# before being passed to the pipeline call.
_IMAGE_INPUT_KEYS = (
    "image",
    "mask_image",
    "control_image",
    "ip_adapter_image",
    "image_2",
)

# Source for the diffusers install used by --remote jobs. While iterating on a
# feature branch, point at the GitHub tarball URL — uv installs it over plain
# HTTP and the container doesn't need ``git``. Once merged, switch back to a
# PyPI release pin. ``--dependencies "diffusers @ ..."`` on the local command
# appends additional dependencies but does not replace this default install.
DIFFUSERS_SOURCE = (
    "diffusers @ https://github.com/huggingface/diffusers/archive/refs/heads/diffuser-cli-for-agent.tar.gz"
)
_DEFAULT_REMOTE_DEPS = (
    DIFFUSERS_SOURCE,
    "accelerate",
    "transformers",
    "safetensors",
    "sentencepiece",  # required by several text-encoder tokenizers (T5, LLaMA, …)
    "ftfy",  # required by older CLIP text-encoder paths
    "kernels",  # required by hub-hosted attention backends (flash_hub, sage_hub, …)
)

# Base container image — provides torch + CUDA so ``uv pip install --system``
# only has to add the small Python deps. cuda12.8 is the highest cuda12.x tag
# below the HF Jobs host driver's CUDA 12.9 max.
_DEFAULT_REMOTE_IMAGE = "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"

# Installed console-script name invoked inside the container after the deps land.
_CONTAINER_CLI_BINARY = "diffusers-cli"

RUN_ID_ENV = "DIFFUSERS_CLI_RUN_ID"

# Namespace keys that control *how* a remote job runs locally, not what runs
# inside the container. They are stripped when forwarding argv to the container.
HF_JOBS_KEYS = frozenset(
    {
        "remote",
        "flavor",
        "timeout",
        "dependencies",
        "namespace",
        "no_wait",
        "poll_interval",
        "func",
        "format",  # top-level --format is a local rendering flag; never forward to the container
    }
)


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------


def _add_loading_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--model", "-m", required=True, help="Model id on the Hugging Face Hub or local path.")
    parser.add_argument("--device", default=None, help="Device to run on (e.g. cpu, cuda, cuda:0, mps).")
    parser.add_argument("--dtype", default="auto", choices=DTYPE_CHOICES, help="Torch dtype for pipeline weights.")
    parser.add_argument("--variant", default=None, help='Optional weight variant (e.g. "fp16").')
    parser.add_argument("--revision", default=None, help="Model revision (branch, tag, or commit SHA).")
    parser.add_argument("--token", default=None, help="Hugging Face token for gated/private models.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom code from the Hub.")
    parser.add_argument(
        "--lora",
        default=None,
        help=(
            "JSON object describing a LoRA adapter to attach after the pipeline loads. "
            'Shape: {"lora_id": "<hub-id-or-path>", "lora_scale": <float>}. '
            'Example: \'{"lora_id": "alvdansen/littletinies", "lora_scale": 0.8}\'.'
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
            "'group' uses pipeline.enable_group_offload(leaf_level, use_stream=True)."
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
        const="{}",
        default=None,
        metavar="JSON",
        help=(
            "torch.compile every denoiser submodule on the pipeline. Accepts an optional JSON "
            'object of kwargs forwarded to ``torch.compile``, e.g. \'{"mode": "max-autotune", '
            '"fullgraph": true}\'. Bare ``--compile`` uses torch defaults. Adds a one-time compilation '
            "cost on the first step but speeds up every subsequent step — worth it for multi-step "
            "generation (50+ steps)."
        ),
    )


def _add_output_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file or directory. Defaults to ./outputs/<task>-<index>.<ext>.",
    )
    parser.add_argument(
        "--push-to",
        default=None,
        help=(
            "Upload the generated files to this HF bucket id after saving (created if missing). "
            "When --remote is set, defaults to <user>/jobs-artifacts."
        ),
    )


def _add_remote_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Submit this command to Hugging Face Jobs instead of running locally.",
    )
    parser.add_argument(
        "--flavor",
        default="a10g-small",
        help="HF Jobs hardware flavor for --remote (e.g. a10g-small, a100-large, cpu-basic).",
    )
    parser.add_argument(
        "--timeout",
        default="10m",
        help="HF Jobs timeout for --remote (e.g. 30m, 2h). Defaults to 10m.",
    )
    parser.add_argument(
        "--dependencies",
        action="append",
        default=None,
        help="Extra pip dependencies for the --remote job. Repeat to add multiple.",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="HF namespace to run the --remote job under (defaults to the current user).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for the --remote job to finish — submit and print the job id.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between job-status polls when waiting for --remote completion.",
    )


# ---------------------------------------------------------------------------
# Pipeline loading + optimization
# ---------------------------------------------------------------------------


def _resolve_dtype(name: str | None):
    if name in (None, "auto"):
        return "auto"
    import torch

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


def _resolve_device(name: str | None) -> str:
    if name:
        return name

    from diffusers.utils.torch_utils import torch_device

    # Under torchrun, LOCAL_RANK identifies this process's assigned GPU. Without this
    # pin every rank falls back to cuda:0 and OOMs as the pipeline replicates onto a
    # single device. Only applies to cuda — torch_device already handles npu/xpu/mps/etc.
    if torch_device == "cuda":
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            import torch

            torch.cuda.set_device(int(local_rank))
            return f"cuda:{local_rank}"
    return torch_device


def _map_to_device(pipeline: Any, args: Namespace, device: str) -> Any:
    """Move the pipeline to ``device``, or hand off to the chosen CPU-offload helper."""
    if args.cpu_offload is None:
        return pipeline.to(device)
    if args.cpu_offload == "model":
        pipeline.enable_model_cpu_offload(device=device)
    elif args.cpu_offload == "group":
        import torch

        pipeline.enable_group_offload(
            onload_device=torch.device(device),
            offload_type="leaf_level",
            use_stream=True,
        )
    return pipeline


def _denoiser(pipeline: Any) -> Any | None:
    """Return the pipeline's denoiser submodule (transformer or unet) or None."""
    for attr in ("transformer", "unet"):
        module = getattr(pipeline, attr, None)
        if module is not None:
            return module
    return None


def _set_attention_backend(pipeline: Any, backend: str) -> None:
    module = _denoiser(pipeline)
    if module is None or not hasattr(module, "set_attention_backend"):
        return
    try:
        module.set_attention_backend(backend)
    except (ValueError, ImportError, RuntimeError) as e:
        raise SystemExit(
            f"Failed to set attention backend {backend!r}: {type(e).__name__}: {e}. "
            f"Allowed backends: {', '.join(ATTENTION_BACKEND_CHOICES)}."
        ) from e


def _enable_context_parallel(pipeline: Any) -> None:
    import torch

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

    from diffusers import ContextParallelConfig

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
        _compile_denoiser(pipeline, args.compile)


def _compile_denoiser(pipeline: Any, compile_spec: str) -> None:
    """Compile every ``transformer*`` and ``unet*`` submodule on the pipeline.

    ``compile_spec`` is the raw JSON string from ``--compile`` (``"{}"`` for bare flag). Decoded into kwargs and
    forwarded verbatim to the compile call.

    Prefers regional compilation via ``module.compile_repeated_blocks(**kwargs)`` — only compiles the repeated inner
    blocks (the bulk of the compute), much faster first-step latency than compiling the whole module. Falls back to
    full ``torch.compile`` if the model doesn't expose ``_repeated_blocks``.
    """
    import torch

    try:
        compile_kwargs = json.loads(compile_spec)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--compile must be valid JSON: {e}") from e
    if not isinstance(compile_kwargs, dict):
        raise SystemExit("--compile must decode to a JSON object.")

    for attr in dir(pipeline):
        if not (attr.startswith("transformer") or attr.startswith("unet")):
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


def _from_pretrained_kwargs(args: Namespace) -> dict[str, Any]:
    dtype = _resolve_dtype(args.dtype)
    kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code, "disable_mmap": True}
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    if args.variant:
        kwargs["variant"] = args.variant
    if args.revision:
        kwargs["revision"] = args.revision
    if args.token:
        kwargs["token"] = args.token
    return kwargs


def _load_pipeline(args: Namespace, modular: bool) -> Any:
    import diffusers

    pipeline_cls = diffusers.ModularPipeline if modular else diffusers.DiffusionPipeline
    pipeline = pipeline_cls.from_pretrained(args.model, **_from_pretrained_kwargs(args))
    if not hasattr(pipeline, "to"):
        return pipeline
    pipeline = _map_to_device(pipeline, args, _resolve_device(args.device))
    _apply_optimizations(pipeline, args)
    _load_lora(pipeline, args)
    return pipeline


def _load_lora(pipeline: Any, args: Namespace) -> None:
    """Attach a LoRA adapter from a JSON spec like ``{"lora_id": "...", "lora_scale": 0.8}``."""
    if not args.lora:
        return
    try:
        spec = json.loads(args.lora)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--lora must be valid JSON: {e}") from e
    if not isinstance(spec, dict):
        raise SystemExit("--lora must decode to a JSON object.")
    lora_id = spec.get("lora_id")
    if not lora_id:
        raise SystemExit("--lora must include a 'lora_id' field.")
    if not hasattr(pipeline, "load_lora_weights"):
        raise SystemExit(f"{type(pipeline).__name__} does not support LoRA loading.")

    pipeline.load_lora_weights(lora_id, adapter_name="default")
    scale = spec.get("lora_scale")
    if scale is not None and hasattr(pipeline, "set_adapters"):
        pipeline.set_adapters(["default"], adapter_weights=[float(scale)])


# ---------------------------------------------------------------------------
# Modular pipeline detection + introspection
# ---------------------------------------------------------------------------


def _is_modular_repo(args: Namespace) -> bool:
    """Detect by trying ``DiffusionPipeline.config_name`` first; modular iff that's absent."""
    from diffusers import DiffusionPipeline

    return try_fetch_config(args, DiffusionPipeline.config_name) is None


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


def _resolve_image_inputs(call_kwargs: dict[str, Any]) -> None:
    """Replace string paths/URLs at known image-input keys with PIL images."""
    for key in _IMAGE_INPUT_KEYS:
        value = call_kwargs.get(key)
        if isinstance(value, str):
            call_kwargs[key] = load_image(value)


def _get_generator(seed: int | None, device: str):
    if seed is None:
        return None
    import torch

    generator_device = "cpu" if device == "mps" else device
    return torch.Generator(device=generator_device).manual_seed(seed)


def _result_to_savable(result: Any) -> Any:
    """Unwrap a pipeline-output object into the raw payload the saver can sniff."""
    if hasattr(result, "images"):
        return result.images
    if hasattr(result, "frames"):
        frames = result.frames
        return frames[0] if isinstance(frames, (list, tuple)) and frames else frames
    if hasattr(result, "audios"):
        return result.audios
    return result


# ---------------------------------------------------------------------------
# Output saving (auto-sniff by type)
# ---------------------------------------------------------------------------


def _default_output_paths(task: str, num: int, explicit: str | None, ext: str) -> list[Path]:
    if explicit is None:
        base = Path(DEFAULT_OUTPUT_DIR)
        base.mkdir(parents=True, exist_ok=True)
        return [base / f"{task}-{i}.{ext}" for i in range(num)]

    p = Path(explicit)
    if explicit.endswith(os.sep) or p.is_dir():
        p.mkdir(parents=True, exist_ok=True)
        return [p / f"{task}-{i}.{ext}" for i in range(num)]

    p.parent.mkdir(parents=True, exist_ok=True)
    if num == 1:
        return [p]
    stem, suffix = p.stem, p.suffix or f".{ext}"
    return [p.with_name(f"{stem}-{i}{suffix}") for i in range(num)]


def _as_pil_list(value: Any):
    try:
        from PIL.Image import Image as PILImage
    except ImportError:
        return None
    if isinstance(value, PILImage):
        return [value]
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, PILImage) for v in value):
        return list(value)
    return None


def _as_frame_sequence(value: Any):
    try:
        from PIL.Image import Image as PILImage
    except ImportError:
        PILImage = None  # type: ignore[assignment]

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        first = value[0]
        if PILImage is not None and isinstance(first, PILImage):
            return list(value)
        try:
            import numpy as np

            if isinstance(first, np.ndarray):
                return list(value)
        except ImportError:
            pass
    return None


def _as_audio_arrays(value: Any):
    try:
        import numpy as np
    except ImportError:
        return None
    if isinstance(value, np.ndarray) and value.ndim <= 2:
        return [value]
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, np.ndarray) for v in value):
        return list(value)
    return None


def _save_audio_arrays(audios, sampling_rate: int, args: Namespace, task: str) -> list[str]:
    """Write each numpy audio array to a 16-bit PCM WAV at ``sampling_rate`` Hz.

    Uses the stdlib ``wave`` module so no scipy dependency is required.
    """
    import wave

    import numpy as np

    paths = _default_output_paths(task, len(audios), args.output, ext="wav")
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


def _save_output(value: Any, args: Namespace, task: str) -> list[str]:
    """Save ``value`` by sniffing its runtime type."""
    pil_images = _as_pil_list(value)
    if pil_images is not None:
        paths = _default_output_paths(task, len(pil_images), args.output, ext="png")
        for img, path in zip(pil_images, paths):
            img.save(path)
        return [str(p) for p in paths]

    frames = _as_frame_sequence(value)
    if frames is not None:
        from diffusers.utils import export_to_video

        path = _default_output_paths(task, 1, args.output, ext="mp4")[0]
        export_to_video(frames, str(path), fps=args.fps)
        return [str(path)]

    audios = _as_audio_arrays(value)
    if audios is not None:
        return _save_audio_arrays(audios, args.sampling_rate or 16000, args, task)

    path = _default_output_paths(task, 1, args.output, ext="json")[0]
    Path(path).write_text(json.dumps(value, default=str, indent=2))
    return [str(path)]


# ---------------------------------------------------------------------------
# Hub bucket upload (--push-to)
# ---------------------------------------------------------------------------


def _push_outputs(args: Namespace, saved_paths: list[str], task: str) -> dict[str, Any] | None:
    """Upload ``saved_paths`` to the ``--push-to`` bucket. Returns a summary or None."""
    if not args.push_to:
        return None

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_bucket(args.push_to, exist_ok=True)

    prefix = os.environ.get(RUN_ID_ENV) or task
    add = [(local, f"{prefix}/{Path(local).name}") for local in saved_paths]
    api.batch_bucket_files(args.push_to, add=add)

    uploaded = [f"hf://buckets/{args.push_to}/{dest}" for _, dest in add]
    return {"bucket_id": args.push_to, "uploaded": uploaded}


# ---------------------------------------------------------------------------
# Remote submission (HF Jobs)
# ---------------------------------------------------------------------------


def _build_task_kwargs(args: Namespace) -> dict[str, Any]:
    """Pick out the kwargs the container should invoke the task with."""
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in HF_JOBS_KEYS or value is None or value is False:
            continue
        out[key] = value
    return out


def _kwargs_to_argv(task: str, task_kwargs: dict[str, Any]) -> list[str]:
    """Render ``task_kwargs`` as the argv list the container's argparse will see."""
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


def _maybe_submit_remote(args: Namespace, task: str) -> bool:
    """If ``--remote`` was set, submit this invocation to HF Jobs and return True."""
    if not args.remote:
        return False

    print(
        f"[diffusers-cli] preparing remote {task!r} job on flavor={args.flavor!r}...",
        file=sys.stderr,
        flush=True,
    )

    import shlex
    import uuid

    from huggingface_hub import HfApi, get_token, run_job

    hf_token = args.token or get_token()
    api = HfApi(token=hf_token)

    if not args.push_to:
        args.push_to = f"{api.whoami()['name']}/jobs-artifacts"

    run_id = uuid.uuid4().hex[:12]

    task_kwargs = _build_task_kwargs(args)
    dependencies = list(_DEFAULT_REMOTE_DEPS)
    if args.dependencies:
        dependencies.extend(args.dependencies)

    secrets = {"HF_TOKEN": hf_token} if hf_token else None
    env = {
        RUN_ID_ENV: run_id,
        "HF_ENABLE_PARALLEL_LOADING": "1",  # thread-pool the safetensors load step
    }

    if Path(args.model).exists():
        print(
            f"[diffusers-cli] WARNING: --model {args.model!r} is a local path; the container can't see it. "
            "Pass a Hub repo id so the job can download it.",
            file=sys.stderr,
            flush=True,
        )

    # Build the in-container shell command: install the small Python deps into the
    # image's system Python (where torch + CUDA already live) via ``uv pip install
    # --system``, then exec the CLI with the same argv. --break-system-packages
    # bypasses PEP 668; safe here because the container is ephemeral.
    # For --context-parallel, wrap with torchrun so torch.distributed initializes
    # across every visible GPU before our generate command runs.
    install_cmd = shlex.join(["uv", "pip", "install", "--system", "--break-system-packages", *dependencies])
    cli_argv = _kwargs_to_argv(task, task_kwargs)
    if args.context_parallel:
        cli_argv = ["torchrun", "--nproc-per-node=gpu", "-m", "diffusers.commands.diffusers_cli", *cli_argv]
    else:
        cli_argv = [_CONTAINER_CLI_BINARY, *cli_argv]
    cli_cmd = shlex.join(cli_argv)
    container_cmd = ["sh", "-c", f"{install_cmd} && {cli_cmd}"]

    job = run_job(
        image=_DEFAULT_REMOTE_IMAGE,
        command=container_cmd,
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=args.namespace,
        secrets=secrets,
        env=env,
        token=hf_token,
    )

    payload: dict[str, Any] = {
        "task": "remote-submit",
        "job_id": getattr(job, "id", None),
        "job_status": str(getattr(job, "status", "")),
        "flavor": args.flavor,
        "push_to": args.push_to,
        "run_id": run_id,
    }

    if args.no_wait:
        _format_result(payload)
        return True

    print(
        f"[diffusers-cli] submitted job {job.id} (run_id={run_id}); "
        f"watch at {getattr(job, 'url', 'https://huggingface.co/jobs')}",
        file=sys.stderr,
        flush=True,
    )
    final_status = _wait_for_job(api, job.id, args.namespace, args.poll_interval)
    payload["job_status"] = final_status
    payload["timing"] = _job_timing(api, job.id, args.namespace)
    payload["outputs"] = _download_job_artifacts(api, args.push_to, run_id, args.output)
    _format_result(payload)
    return True


def _job_timing(api: Any, job_id: str, namespace: str | None) -> dict[str, float | None]:
    """Return queue/run/total wallclock seconds for ``job_id`` from inspect_job timestamps.

    inspect_job sometimes returns finished_at=None for a few seconds after the container exits while HF Jobs propagates
    the terminal state; retry briefly so we don't miss run/total.
    """
    import time

    info = api.inspect_job(job_id=job_id, namespace=namespace)
    for _ in range(5):
        if info.finished_at is not None:
            break
        time.sleep(1.0)
        info = api.inspect_job(job_id=job_id, namespace=namespace)

    def _delta(start, end) -> float | None:
        return (end - start).total_seconds() if (start is not None and end is not None) else None

    timing = {
        "queued_seconds": _delta(info.created_at, info.started_at),
        "run_seconds": _delta(info.started_at, info.finished_at),
        "total_seconds": _delta(info.created_at, info.finished_at),
    }
    parts = [f"{k.replace('_seconds', '')}={v:.1f}s" for k, v in timing.items() if v is not None]
    if parts:
        print(f"[diffusers-cli] timing: {' '.join(parts)}", file=sys.stderr, flush=True)
    return timing


def _wait_for_job(api: Any, job_id: str, namespace: str | None, poll_interval: float) -> str:
    """Stream container logs to stderr until the job terminates; return the final stage."""
    fetch = getattr(api, "fetch_job_logs", None)
    if fetch is not None:
        try:
            for line in fetch(job_id=job_id, namespace=namespace, follow=True):
                print(line, file=sys.stderr, flush=True)
        except TypeError:
            return _poll_for_job(api, job_id, namespace, poll_interval)
        info = api.inspect_job(job_id=job_id, namespace=namespace)
        return str(info.status.stage) if info.status else "UNKNOWN"
    return _poll_for_job(api, job_id, namespace, poll_interval)


def _poll_for_job(api: Any, job_id: str, namespace: str | None, poll_interval: float) -> str:
    """Heartbeat-style fallback when ``fetch_job_logs`` isn't available."""
    import time

    terminal = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}
    last_stage: str | None = None
    while True:
        info = api.inspect_job(job_id=job_id, namespace=namespace)
        stage = str(info.status.stage) if info.status else "UNKNOWN"
        if stage != last_stage:
            if last_stage is not None:
                print("", file=sys.stderr, flush=True)
            print(f"[diffusers-cli] job {job_id}: {stage}", file=sys.stderr, flush=True)
            last_stage = stage
        else:
            print(".", end="", file=sys.stderr, flush=True)
        if stage in terminal:
            print("", file=sys.stderr, flush=True)
            return stage
        time.sleep(poll_interval)


def _download_job_artifacts(api: Any, bucket_id: str, run_id: str, output: str | None) -> list[str]:
    """Download every file under ``<run_id>/`` from ``bucket_id`` into a local directory."""
    from huggingface_hub import BucketFile

    local_dir = Path(output) if output else Path(DEFAULT_OUTPUT_DIR)
    local_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[Any, Path]] = []
    for entry in api.list_bucket_tree(bucket_id, prefix=f"{run_id}/", recursive=True):
        if not isinstance(entry, BucketFile):
            continue
        pairs.append((entry, local_dir / Path(entry.path).name))

    if not pairs:
        return []
    api.download_bucket_files(bucket_id, files=pairs)
    return [str(local) for _, local in pairs]


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def _format_result(payload: dict[str, Any]) -> None:
    """Route the result payload through the output sink."""
    out.result(payload.get("task", "done"), **payload)


# ---------------------------------------------------------------------------
# Subcommand
# ---------------------------------------------------------------------------


class GenerateCommand(BaseDiffusersCLICommand):
    task = "generate"

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        from argparse import RawDescriptionHelpFormatter

        epilog = (
            "Examples\n"
            "  $ diffusers-cli generate -m black-forest-labs/FLUX.1-dev --dtype bf16 \\\n"
            '      --pipeline-kwargs \'{"prompt": "a cat on the moon"}\'\n'
            "  $ diffusers-cli generate -m black-forest-labs/FLUX.1-dev --dtype bf16 \\\n"
            '      --pipeline-kwargs \'{"prompt": "make the fur grey", "image": "https://example.com/cat.png"}\'\n'
            "  $ diffusers-cli generate -m black-forest-labs/FLUX.1-dev --dtype bf16 \\\n"
            '      --pipeline-kwargs \'{"prompt": "a tiny cat"}\' \\\n'
            '      --lora \'{"lora_id": "alvdansen/littletinies", "lora_scale": 0.8}\'\n'
            "  $ diffusers-cli generate -m black-forest-labs/FLUX.1-dev --dtype bf16 \\\n"
            '      --pipeline-kwargs \'{"prompt": "a cat"}\' --remote --flavor a100-large\n'
            "  $ diffusers-cli generate -m black-forest-labs/FLUX.1-dev --dtype bf16 --context-parallel \\\n"
            '      --pipeline-kwargs \'{"prompt": "a cat"}\' --remote --flavor 4xa100-large\n'
            "\n"
            "Learn more\n"
            "  Use `diffusers-cli <command> --help` for more information about a command.\n"
            "  Read the documentation at https://huggingface.co/docs/diffusers\n"
        )

        parser: ArgumentParser = subparsers.add_parser(
            "generate",
            help="Run any diffusers pipeline locally or remotely with HF Jobs.",
            usage="\n  diffusers-cli generate [options]",
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
                f"image-input keys ({', '.join(_IMAGE_INPUT_KEYS)}) are auto-loaded as PIL images."
            ),
        )
        parser.add_argument(
            "--output-key",
            default=None,
            help="For modular pipelines: name of the intermediate to extract (passed as `output=` to the call).",
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
        parser.set_defaults(func=GenerateCommand)

    def __init__(self, args: Namespace):
        self.args = args

    def run(self) -> None:
        is_modular = _is_modular_repo(self.args)

        if _maybe_submit_remote(self.args, self.task):
            return

        pipeline = _load_pipeline(self.args, modular=is_modular)

        call_kwargs = _parse_pipeline_kwargs(self.args.pipeline_kwargs)
        _resolve_image_inputs(call_kwargs)

        if self.args.output_key is not None:
            call_kwargs["output"] = self.args.output_key

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
                savable = result if is_modular else _result_to_savable(result)
                saved = _save_output(savable, self.args, self.task)
                pushed = _push_outputs(self.args, saved, self.task)

                _format_result(
                    {
                        "task": self.task,
                        "model": self.args.model,
                        "device": device,
                        "pipeline_class": type(pipeline).__name__,
                        "modular": is_modular,
                        "outputs": saved,
                        "pushed": pushed,
                        "seed": self.args.seed,
                        "output_key": self.args.output_key,
                    },
                )
        finally:
            import torch

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
