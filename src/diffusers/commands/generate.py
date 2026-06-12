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
from typing import Any, Optional

from diffusers.utils import load_image

from . import BaseDiffusersCLICommand


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = "outputs"
DTYPE_CHOICES = ("auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32")
CPU_OFFLOAD_CHOICES = ("model", "group")
ATTENTION_BACKEND_CHOICES = (
    "default",
    "flash_hub",
    "flash_varlen_hub",
    "flash_4_hub",
    "sage_hub",
)

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
    {"remote", "flavor", "timeout", "dependencies", "namespace", "no_wait", "poll_interval", "func"}
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
            "Requires launching the CLI under torchrun with ≥2 GPUs."
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
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON summary on stdout.")


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
    parser.add_argument("--timeout", default=None, help="HF Jobs timeout for --remote (e.g. 30m, 2h).")
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


def _resolve_dtype(name: Optional[str]):
    if name in (None, "auto"):
        return "auto"
    import torch

    mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if name not in mapping:
        raise ValueError(f"Unknown dtype: {name}")
    return mapping[name]


def _resolve_device(name: Optional[str]) -> str:
    if name:
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
            use_stream=device.startswith("cuda"),
        )
    return pipeline


def _set_attention_backend(pipeline: Any, backend: str) -> None:
    for attr in ("transformer", "unet"):
        module = getattr(pipeline, attr, None)
        if module is not None and hasattr(module, "set_attention_backend"):
            try:
                module.set_attention_backend(backend)
            except (ValueError, ImportError, RuntimeError):
                pass
            return


def _enable_context_parallel(pipeline: Any) -> None:
    import torch

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise SystemExit(
            "--context-parallel requires torch.distributed to be initialized. "
            "Launch the CLI under torchrun, e.g.: "
            "`torchrun --nproc-per-node=N -m diffusers.commands.diffusers_cli generate ...`."
        )

    from diffusers import ContextParallelConfig

    cfg = ContextParallelConfig(
        ulysses_degree=torch.distributed.get_world_size(),
        ring_degree=1,
        ulysses_anything=True,
    )
    for attr in ("transformer", "unet"):
        module = getattr(pipeline, attr, None)
        if module is not None and hasattr(module, "enable_parallelism"):
            module.enable_parallelism(config=cfg)
            return


def _apply_optimizations(pipeline: Any, args: Namespace) -> None:
    """Apply VAE tiling/slicing, attention backend, and context-parallel toggles."""
    if args.vae_tiling and hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
    if args.vae_slicing and hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if args.attention_backend != "default":
        _set_attention_backend(pipeline, args.attention_backend)
    if args.context_parallel:
        _enable_context_parallel(pipeline)


def _from_pretrained_kwargs(args: Namespace) -> dict[str, Any]:
    dtype = _resolve_dtype(args.dtype)
    # disable_mmap: mmap-faults over a network-mounted volume trigger one round-trip per page;
    # a sequential read is dramatically faster than the random-access mmap pattern.
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
    return pipeline


# ---------------------------------------------------------------------------
# Modular pipeline detection + introspection
# ---------------------------------------------------------------------------


def _try_fetch_config(args: Namespace, filename: str) -> Optional[str]:
    """Try to resolve ``filename`` for ``args.model`` (local path or Hub repo). None if absent."""
    local = Path(args.model)
    if local.exists():
        candidate = local / filename
        return str(candidate) if candidate.exists() else None

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

    try:
        return hf_hub_download(args.model, filename, revision=args.revision, token=args.token)
    except (EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError):
        return None


def _is_modular_repo(args: Namespace) -> bool:
    """Detect by trying ``DiffusionPipeline.config_name`` first; modular iff that's absent."""
    from diffusers import DiffusionPipeline

    return _try_fetch_config(args, DiffusionPipeline.config_name) is None


def _describe(args: Namespace) -> None:
    """Print the pipeline's input schema.

    Tries ``DiffusionPipeline.config_name`` (= ``model_index.json``) first; if present, introspects the declared
    pipeline class's ``__call__`` signature. Otherwise falls back to ``ModularPipelineBlocks.from_pretrained`` and
    reads the block-declared ``inputs``. No weights downloaded either way.
    """
    import inspect

    import diffusers

    standard_index = _try_fetch_config(args, diffusers.DiffusionPipeline.config_name)

    if standard_index is not None:
        with open(standard_index) as f:
            index = json.load(f)
        class_name = index.get("_class_name")
        pipeline_cls = getattr(diffusers, class_name, None)
        if pipeline_cls is None:
            raise SystemExit(
                f"Pipeline class {class_name!r} declared in {diffusers.DiffusionPipeline.config_name} "
                "is not exported by the installed diffusers."
            )
        sig = inspect.signature(pipeline_cls.__call__)
        descriptions = _parse_docstring_args(pipeline_cls.__call__.__doc__) if args.verbose else {}
        schema: list[dict[str, Any]] = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            has_default = param.default is not inspect.Parameter.empty
            schema.append(
                {
                    "name": name,
                    "type_hint": str(param.annotation) if param.annotation is not inspect.Parameter.empty else None,
                    "default": param.default if has_default else None,
                    "required": not has_default,
                    "description": descriptions.get(name, ""),
                }
            )
    else:
        kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
        if args.revision:
            kwargs["revision"] = args.revision
        if args.token:
            kwargs["token"] = args.token
        try:
            blocks = diffusers.ModularPipelineBlocks.from_pretrained(args.model, **kwargs)
        except Exception as e:
            raise SystemExit(
                f"Could not describe {args.model!r}: no {diffusers.DiffusionPipeline.config_name} and "
                f"loading as a modular pipeline failed ({type(e).__name__}: {e}). "
                "Is this a diffusers pipeline repo? Pass --trust-remote-code if it ships custom block code."
            ) from e
        class_name = type(blocks).__name__
        schema = [
            {
                "name": p.name,
                "type_hint": str(p.type_hint) if p.type_hint is not None else None,
                "default": p.default,
                "required": p.required,
                "description": p.description,
            }
            for p in blocks.inputs
        ]

    if args.json:
        payload = {
            "task": "describe",
            "model": args.model,
            "pipeline_class": class_name,
            "inputs": schema,
        }
        json.dump(payload, sys.stdout, default=str)
        sys.stdout.write("\n")
        return

    _print_schema(class_name, args.model, schema)


def _parse_docstring_args(docstring: Optional[str]) -> dict[str, str]:
    """Extract per-argument descriptions from a Google-style ``Args:`` block.

    Returns a ``{name: description}`` mapping. Best-effort — unrecognised formats just yield an empty dict rather than
    raising.
    """
    if not docstring:
        return {}

    import re

    lines = docstring.expandtabs().splitlines()
    start = None
    section_indent = 0
    for i, line in enumerate(lines):
        if line.strip() in ("Args:", "Arguments:", "Parameters:"):
            start = i + 1
            section_indent = len(line) - len(line.lstrip())
            break
    if start is None:
        return {}

    descriptions: dict[str, str] = {}
    current_name: Optional[str] = None
    current_lines: list[str] = []
    arg_indent: Optional[int] = None
    name_pattern = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:?\s*(.*)$")

    def _flush() -> None:
        if current_name and current_lines:
            descriptions[current_name] = " ".join(s.strip() for s in current_lines).strip()

    for line in lines[start:]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        # A new top-level section ends the Args block.
        if indent <= section_indent and line.strip().endswith(":"):
            break
        if arg_indent is None:
            arg_indent = indent
        if indent == arg_indent:
            _flush()
            current_lines = []
            match = name_pattern.match(line.strip())
            if match:
                current_name = match.group(1)
                tail = match.group(2).strip()
                if tail:
                    current_lines.append(tail)
            else:
                current_name = None
        elif current_name is not None and indent > arg_indent:
            current_lines.append(line.strip())
    _flush()
    return descriptions


def _print_schema(class_name: str, model: str, schema: list[dict[str, Any]]) -> None:
    print(f"{class_name} ({model}) inputs:")
    for entry in schema:
        tag = "required" if entry["required"] else f"optional, default={entry['default']!r}"
        print(f"  {entry['name']}  ({tag})")
        if entry["type_hint"]:
            print(f"    type: {entry['type_hint']}")
        if entry["description"]:
            print(f"    desc: {entry['description']}")


# ---------------------------------------------------------------------------
# Pipeline call helpers
# ---------------------------------------------------------------------------


def _parse_pipeline_kwargs(raw: Optional[str]) -> dict[str, Any]:
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


def _get_generator(seed: Optional[int], device: str):
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


def _default_output_paths(task: str, num: int, explicit: Optional[str], ext: str) -> list[Path]:
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
    """Write each numpy audio array to a 16-bit PCM WAV at ``sampling_rate`` Hz."""
    import numpy as np
    from scipy.io.wavfile import write as wavfile_write

    paths = _default_output_paths(task, len(audios), args.output, ext="wav")
    saved: list[str] = []
    for audio, path in zip(audios, paths):
        data = np.asarray(audio)
        if data.dtype.kind == "f":
            data = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
        if data.ndim > 1 and data.shape[0] < data.shape[-1]:
            data = data.T  # (channels, samples) → (samples, channels) for scipy.
        wavfile_write(str(path), sampling_rate, data)
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


def _push_outputs(args: Namespace, saved_paths: list[str], task: str) -> Optional[dict[str, Any]]:
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
    install_cmd = shlex.join(["uv", "pip", "install", "--system", "--break-system-packages", *dependencies])
    cli_cmd = shlex.join([_CONTAINER_CLI_BINARY, *_kwargs_to_argv(task, task_kwargs)])
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
        _format_result(args, payload)
        return True

    print(
        f"[diffusers-cli] submitted job {job.id} (run_id={run_id}); "
        f"watch at {getattr(job, 'url', 'https://huggingface.co/jobs')}",
        file=sys.stderr,
        flush=True,
    )
    final_status = _wait_for_job(api, job.id, args.namespace, args.poll_interval)
    payload["job_status"] = final_status
    payload["outputs"] = _download_job_artifacts(api, args.push_to, run_id, args.output)
    _format_result(args, payload)
    return True


def _wait_for_job(api: Any, job_id: str, namespace: Optional[str], poll_interval: float) -> str:
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


def _poll_for_job(api: Any, job_id: str, namespace: Optional[str], poll_interval: float) -> str:
    """Heartbeat-style fallback when ``fetch_job_logs`` isn't available."""
    import time

    terminal = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}
    last_stage: Optional[str] = None
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


def _download_job_artifacts(api: Any, bucket_id: str, run_id: str, output: Optional[str]) -> list[str]:
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


def _format_result(args: Namespace, payload: dict[str, Any]) -> None:
    """Print either a human-friendly summary or JSON, depending on --json."""
    if args.json:
        json.dump(payload, sys.stdout, default=str)
        sys.stdout.write("\n")
        return

    outputs = payload.get("outputs", [])
    if outputs:
        for path in outputs:
            print(path)
    else:
        print(payload)


# ---------------------------------------------------------------------------
# Subcommand
# ---------------------------------------------------------------------------


class GenerateCommand(BaseDiffusersCLICommand):
    task = "generate"

    @staticmethod
    def register_subcommand(subparsers: _SubParsersAction) -> None:
        parser: ArgumentParser = subparsers.add_parser(
            "generate",
            help="Run any diffusers pipeline (standard or modular) by forwarding --pipeline-kwargs verbatim.",
            usage="\n  diffusers-cli generate [options]",
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

        result = pipeline(**call_kwargs)
        savable = result if is_modular else _result_to_savable(result)
        saved = _save_output(savable, self.args, self.task)
        pushed = _push_outputs(self.args, saved, self.task)

        _format_result(
            self.args,
            {
                "task": self.task,
                "model": self.args.model,
                "device": pipeline.device.type if hasattr(pipeline, "device") else None,
                "pipeline_class": type(pipeline).__name__,
                "modular": is_modular,
                "outputs": saved,
                "pushed": pushed,
                "seed": self.args.seed,
                "output_key": self.args.output_key,
            },
        )
