# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""Shared helpers for the agentic CLI surface.

These utilities are intentionally small and dependency-light. Each diffusers
agentic subcommand should be able to be read end-to-end by an agent without
needing to follow many layers of indirection.
"""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Optional


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


def add_loading_arguments(parser: ArgumentParser) -> None:
    """Arguments shared by every inference subcommand."""
    parser.add_argument("--model", "-m", required=True, help="Model id on the Hugging Face Hub or local path.")
    parser.add_argument("--device", default=None, help="Device to run on (e.g. cpu, cuda, cuda:0, mps).")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=DTYPE_CHOICES,
        help="Torch dtype for pipeline weights.",
    )
    parser.add_argument("--variant", default=None, help='Optional weight variant (e.g. "fp16").')
    parser.add_argument("--revision", default=None, help="Model revision (branch, tag, or commit SHA).")
    parser.add_argument("--token", default=None, help="Hugging Face token for gated/private models.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom code from the Hub.")


def add_optimization_arguments(parser: ArgumentParser) -> None:
    """Optional pipeline-optimization flags. All default to off."""
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
            "Only Hub-hosted kernels are exposed — they auto-download on first "
            "use and avoid a local install. 'default' leaves the backend untouched."
        ),
    )
    parser.add_argument("--vae-tiling", action="store_true", help="Enable VAE tiling (lower peak VRAM).")
    parser.add_argument("--vae-slicing", action="store_true", help="Enable VAE slicing (lower peak VRAM).")
    parser.add_argument(
        "--context-parallel",
        action="store_true",
        help=(
            "Enable Ulysses-style context parallelism (ulysses_anything mode, supports arbitrary "
            "sequence lengths). Requires launching the CLI under torchrun with ≥2 GPUs."
        ),
    )


def add_generation_arguments(parser: ArgumentParser) -> None:
    """Arguments shared by image/video generation subcommands."""
    parser.add_argument("--prompt", "-p", default=None, help="Text prompt.")
    parser.add_argument("--negative-prompt", default=None, help="Negative text prompt.")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Number of denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Classifier-free guidance scale.")
    parser.add_argument("--height", type=int, default=None, help="Output height in pixels.")
    parser.add_argument("--width", type=int, default=None, help="Output width in pixels.")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--pipeline-kwargs",
        default=None,
        help="JSON object of extra kwargs forwarded to the pipeline call.",
    )


def add_output_arguments(parser: ArgumentParser) -> None:
    """Output formatting arguments."""
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
            "Upload the generated files to this HF bucket id after saving "
            "(created if missing). When --remote is set, defaults to "
            "<user>/jobs-artifacts; remote runs always write to that bucket "
            "and fetch the results back locally."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON summary on stdout.")


def add_remote_arguments(parser: ArgumentParser) -> None:
    """Optional HF Jobs arguments — works on every inference subcommand."""
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
        default=None,
        help="HF Jobs timeout for --remote (e.g. 30m, 2h).",
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
        help=(
            "Don't wait for the --remote job to finish — submit and print the job id. "
            "Default behaviour is to poll until completion and download outputs locally."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between job-status polls when waiting for --remote completion.",
    )


def resolve_dtype(name: Optional[str]):
    """Map a CLI dtype string to a torch dtype.

    Returns ``"auto"`` when the user wants diffusers to pick.
    """
    if name in (None, "auto"):
        return "auto"

    import torch

    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unknown dtype: {name}")
    return mapping[name]


def resolve_device(name: Optional[str]) -> str:
    """Pick a device, defaulting to the best available one."""
    if name:
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_pipeline(args: Namespace, pipeline_cls_name: str) -> Any:
    """Load a diffusers pipeline class by name and move it to the chosen device.

    ``pipeline_cls_name`` can be any class exported from ``diffusers`` —
    typically one of ``AutoPipelineForText2Image``, ``AutoPipelineForImage2Image``,
    ``AutoPipelineForInpainting``, or ``DiffusionPipeline`` for video/audio.
    """
    import diffusers

    pipeline_cls = getattr(diffusers, pipeline_cls_name)
    from_pretrained_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.variant:
        from_pretrained_kwargs["variant"] = args.variant
    if args.revision:
        from_pretrained_kwargs["revision"] = args.revision
    if args.token:
        from_pretrained_kwargs["token"] = args.token

    pipeline = pipeline_cls.from_pretrained(args.model, **from_pretrained_kwargs)
    pipeline = map_to_device(pipeline, args, resolve_device(args.device))
    if args.vae_tiling and hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
    if args.vae_slicing and hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if args.attention_backend != "default":
        _set_attention_backend(pipeline, args.attention_backend)
    if args.context_parallel:
        _enable_context_parallel(pipeline)
    return pipeline


def map_to_device(pipeline: Any, args: Namespace, device: str) -> Any:
    """Get the pipeline ready to run on ``device``.

    Calls ``.to(device)`` by default; when ``--cpu-offload`` is set the chosen
    offload helper (``model``, ``sequential``, or ``group``) handles placement instead.
    """
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


def _enable_context_parallel(pipeline: Any) -> None:
    """Enable Ulysses-style context-parallel inference on the transformer/UNet."""
    import torch

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise SystemExit(
            "--context-parallel requires torch.distributed to be initialized. "
            "Launch the CLI under torchrun, e.g.: "
            "`torchrun --nproc-per-node=N -m diffusers.commands.diffusers_cli <task> ...`."
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


def _set_attention_backend(pipeline: Any, backend: str) -> None:
    for attr in ("transformer", "unet"):
        module = getattr(pipeline, attr, None)
        if module is not None and hasattr(module, "set_attention_backend"):
            try:
                module.set_attention_backend(backend)
            except (ValueError, ImportError, RuntimeError):
                pass
            return


def get_generator(seed: Optional[int], device: str):
    if seed is None:
        return None
    import torch

    generator_device = "cpu" if device == "mps" else device
    return torch.Generator(device=generator_device).manual_seed(seed)


def parse_pipeline_kwargs(raw: Optional[str]) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--pipeline-kwargs must be valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise SystemExit("--pipeline-kwargs must decode to a JSON object.")
    return parsed


def default_output_paths(task: str, num: int, explicit: Optional[str], ext: str = "png") -> list[Path]:
    """Resolve output file paths for ``num`` generated artifacts.

    - If ``explicit`` is a directory (or ends with /), write into it.
    - If ``explicit`` is a file and ``num == 1``, write to that file.
    - If ``explicit`` is a file template and ``num > 1``, append ``-<i>`` before the suffix.
    - Otherwise default to ``./outputs/<task>-<i>.<ext>``.
    """
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


# Source for the diffusers install used by --remote jobs. While iterating on a
# feature branch, point at the branch URL; once merged, switch back to a release
# pin. ``--dependencies "diffusers @ git+..."`` on the local command appends
# additional dependencies but does not replace this default install.
DIFFUSERS_SOURCE = "diffusers @ git+https://github.com/huggingface/diffusers@diffuser-cli-for-agent"
_DEFAULT_REMOTE_DEPS = (DIFFUSERS_SOURCE, "accelerate", "transformers", "safetensors")

# Entry point for ``uv run`` inside the container. ``uv run`` accepts a file path,
# URL, or *command*; passing the ``diffusers-cli`` console script name makes UV
# install the deps above (which register the entry point) and then exec the CLI.
_UV_RUNNER_SCRIPT = "diffusers-cli"


RUN_ID_ENV = "DIFFUSERS_CLI_RUN_ID"

# Namespace keys that control *how* a remote job runs locally, not what runs
# inside the container. They are stripped when forwarding argv to the container.
HF_JOBS_KEYS = frozenset(
    {"remote", "flavor", "timeout", "dependencies", "namespace", "no_wait", "poll_interval", "func"}
)


def _rewrite_model_arg(forwarded: list[str], new_path: str) -> list[str]:
    """Return a copy of ``forwarded`` with the ``--model`` value replaced by ``new_path``."""
    out = list(forwarded)
    for i, token in enumerate(out):
        if token in ("--model", "-m") and i + 1 < len(out):
            out[i + 1] = new_path
            return out
    return out


def _forward_args(args: Namespace, task: str) -> list[str]:
    """Reconstruct argv for the remote container from a parsed Namespace.

    Skips the local-only job-control keys above. Boolean flags are emitted
    only when True. List values become repeated ``--flag value`` pairs.
    """
    out: list[str] = [task]
    for key, value in vars(args).items():
        if key in HF_JOBS_KEYS:
            continue
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            out.append(flag)
        elif isinstance(value, list):
            for item in value:
                out.extend([flag, str(item)])
        else:
            out.extend([flag, str(value)])
    return out


def maybe_submit_remote(args: Namespace, task: str) -> bool:
    """If ``--remote`` was set, submit this invocation to HF Jobs and return True.

    The local ``run()`` should bail immediately when this returns True.

    Auto-defaults ``--push-to`` to ``<user>/jobs-artifacts`` so the remote
    container has somewhere to write before tear-down. By default, polls
    the job until completion and downloads the artifacts back to the local
    output directory; pass ``--no-wait`` to fire-and-forget.
    """
    if not args.remote:
        return False

    import uuid

    from huggingface_hub import HfApi, get_token, run_uv_job

    try:
        from huggingface_hub import Volume
    except ImportError:
        Volume = None

    hf_token = args.token or get_token()
    api = HfApi(token=hf_token)

    if not args.push_to:
        args.push_to = f"{api.whoami()['name']}/jobs-artifacts"

    run_id = uuid.uuid4().hex[:12]

    forwarded = _forward_args(args, task)
    dependencies = list(_DEFAULT_REMOTE_DEPS)
    if args.dependencies:
        dependencies.extend(args.dependencies)

    secrets = {"HF_TOKEN": hf_token} if hf_token else None
    env = {
        RUN_ID_ENV: run_id,
        "HF_ENABLE_PARALLEL_LOADING": "1",  # thread-pool the safetensors load step
    }

    # Mount the model repo into the job's filesystem so the container reads it
    # from local disk instead of downloading on every run. Requires
    # huggingface_hub >= 1.16. Falls back to the download path otherwise.
    run_uv_job_kwargs: dict[str, Any] = dict(
        script=_UV_RUNNER_SCRIPT,
        script_args=forwarded,
        dependencies=dependencies,
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=args.namespace,
        secrets=secrets,
        env=env,
        token=hf_token,
    )
    if Volume is not None and not Path(args.model).exists():
        mount_path = "/model"
        run_uv_job_kwargs["volumes"] = [
            Volume(type="model", source=args.model, mount_path=mount_path)
        ]
        run_uv_job_kwargs["script_args"] = _rewrite_model_arg(forwarded, mount_path)

    job = run_uv_job(**run_uv_job_kwargs)

    payload: dict[str, Any] = {
        "task": "remote-submit",
        "job_id": getattr(job, "id", None),
        "job_status": str(getattr(job, "status", "")),
        "flavor": args.flavor,
        "push_to": args.push_to,
        "run_id": run_id,
    }

    if args.no_wait:
        format_result(args, payload)
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
    format_result(args, payload)
    return True


def _wait_for_job(api: Any, job_id: str, namespace: Optional[str], poll_interval: float) -> str:
    """Poll ``inspect_job`` until the job reaches a terminal stage; return that stage as a string.

    Prints a heartbeat each poll and a labelled line on every stage transition so
    the local terminal isn't silent for the multi-minute install/download/run
    window of a remote inference job.
    """
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
    """Download every file under ``<run_id>/`` from ``bucket_id`` to a local directory.

    ``output`` is always treated as a directory (created if missing) — remote
    runs produce many files, so a file-path target wouldn't make sense.
    """
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


def push_outputs(args: Namespace, saved_paths: list[str], task: str) -> Optional[dict[str, Any]]:
    """Upload ``saved_paths`` to the ``--push-to`` bucket, returning a summary.

    Returns None when ``--push-to`` is unset. Creates the bucket if needed.
    When ``DIFFUSERS_CLI_RUN_ID`` is set (i.e. we're inside a remote job),
    files land under ``<run_id>/`` so the local side can isolate this run's
    output; otherwise they land under ``<task>/``.
    """
    if not args.push_to:
        return None
    target = args.push_to

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_bucket(target, exist_ok=True)

    prefix = os.environ.get(RUN_ID_ENV) or task
    add = [(local, f"{prefix}/{Path(local).name}") for local in saved_paths]
    api.batch_bucket_files(target, add=add)

    uploaded = [f"hf://buckets/{target}/{dest}" for _, dest in add]
    return {"bucket_id": target, "uploaded": uploaded}


def format_result(args: Namespace, payload: dict[str, Any]) -> None:
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
