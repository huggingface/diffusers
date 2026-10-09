"""
Benchmark end-to-end pipeline optimization recipes and emit JSON + Markdown reports.

Example usage:
    python examples/profiling/profile_pipeline_recipes.py --pipeline flux
    python examples/profiling/profile_pipeline_recipes.py --pipeline wan --full_decode
    python examples/profiling/profile_pipeline_recipes.py \
        --pipeline qwenimage \
        --recipe baseline \
        --recipe layerwise_casting+attention:_native_flash \
        --recipe compile+attention:sage
"""

import argparse
import copy
import gc
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import torch

try:
    from .pipeline_registry import build_registry
except ImportError:
    from pipeline_registry import build_registry


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ACTION_ORDER = {
    "baseline": 0,
    "layerwise_casting": 1,
    "attention": 2,
    "vae_tiling": 3,
    "channels_last": 4,
    "model_cpu_offload": 5,
    "group_offload_leaf": 6,
    "compile": 7,
}
DEFAULT_RECIPES = [
    "baseline",
    "model_cpu_offload",
    "group_offload_leaf",
    "layerwise_casting",
    "compile",
]


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()


@dataclass
class RecipeBenchmarkResult:
    pipeline: str
    recipe: str
    status: str
    mean_ms: float | None
    std_ms: float | None
    peak_vram_gb: float | None
    num_runs: int
    num_warmups: int
    notes: str


def parse_recipe(recipe: str) -> list[tuple[str, str | None]]:
    actions = []
    for raw_token in recipe.split("+"):
        token = raw_token.strip()
        if not token:
            continue
        if ":" in token:
            name, value = token.split(":", 1)
        else:
            name, value = token, None
        if name not in ACTION_ORDER:
            raise ValueError(f"Unknown recipe action '{name}' in recipe '{recipe}'")
        actions.append((name, value))

    if not actions:
        raise ValueError("Recipe cannot be empty")

    deduped = []
    seen = set()
    for name, value in sorted(actions, key=lambda item: ACTION_ORDER[item[0]]):
        key = (name, value)
        if key not in seen and name != "baseline":
            deduped.append((name, value))
            seen.add(key)

    return deduped or [("baseline", None)]


def benchmark_pipeline_call(pipe, call_kwargs, num_runs: int, num_warmups: int) -> tuple[float, float, float]:
    for _ in range(num_warmups):
        pipe(**call_kwargs)
        torch.cuda.synchronize()

    times = []
    peak_memories = []

    for _ in range(num_runs):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        pipe(**call_kwargs)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))
        peak_memories.append(torch.cuda.max_memory_allocated() / (1024**3))

    mean_ms = sum(times) / len(times)
    variance = sum((t - mean_ms) ** 2 for t in times) / len(times)
    return mean_ms, variance**0.5, max(peak_memories)


def resolve_vae_tiling(pipe):
    if hasattr(pipe, "enable_vae_tiling"):
        return pipe.enable_vae_tiling
    vae = getattr(pipe, "vae", None)
    if vae is not None and hasattr(vae, "enable_tiling"):
        return vae.enable_tiling
    return None


def apply_recipe(pipe, actions, *, call_kwargs, compile_kwargs, compile_regional):
    notes = []
    uses_offloading = False

    for action_name, action_value in actions:
        if action_name == "baseline":
            continue

        if action_name == "layerwise_casting":
            transformer = getattr(pipe, "transformer", None)
            if transformer is None or not hasattr(transformer, "enable_layerwise_casting"):
                raise RuntimeError("layerwise_casting requires pipe.transformer.enable_layerwise_casting()")
            compute_dtype = getattr(transformer, "dtype", None) or getattr(pipe, "dtype", None)
            if compute_dtype is None:
                raise RuntimeError("layerwise_casting could not infer a compute dtype for this transformer")
            transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=compute_dtype,
            )
            continue

        if action_name == "attention":
            backend = action_value or "native"
            transformer = getattr(pipe, "transformer", None)
            if transformer is None or not hasattr(transformer, "set_attention_backend"):
                raise RuntimeError("attention backend switching requires pipe.transformer.set_attention_backend()")
            transformer.set_attention_backend(backend)
            notes.append(f"attention_backend={backend}")
            continue

        if action_name == "vae_tiling":
            if call_kwargs.get("output_type") == "latent":
                notes.append("vae_tiling requested with latent output; VAE decode is skipped")
                continue
            enable_vae_tiling = resolve_vae_tiling(pipe)
            if enable_vae_tiling is None:
                raise RuntimeError("vae_tiling is not supported by this pipeline")
            enable_vae_tiling()
            continue

        if action_name == "channels_last":
            if hasattr(pipe, "unet") and pipe.unet is not None:
                pipe.unet.to(memory_format=torch.channels_last)
                continue
            raise RuntimeError("channels_last is only supported for pipelines exposing pipe.unet")

        if action_name == "model_cpu_offload":
            pipe.enable_model_cpu_offload()
            uses_offloading = True
            continue

        if action_name == "group_offload_leaf":
            pipe.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="leaf_level",
            )
            uses_offloading = True
            continue

    if not uses_offloading:
        pipe.to("cuda")

    for action_name, action_value in actions:
        if action_name != "compile":
            continue
        transformer = getattr(pipe, "transformer", None)
        if transformer is None:
            raise RuntimeError("compile requires pipe.transformer")
        if compile_regional and hasattr(transformer, "compile_repeated_blocks"):
            transformer.compile_repeated_blocks(**compile_kwargs)
            notes.append("compile=regional")
        elif hasattr(transformer, "compile"):
            transformer.compile(**compile_kwargs)
            notes.append("compile=full")
        else:
            raise RuntimeError("compile is not supported by this transformer")

    return notes


def load_pipeline(config):
    pipe = config.pipeline_cls.from_pretrained(**config.pipeline_init_kwargs)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_recipe(config, recipe: str, *, num_runs: int, num_warmups: int, compile_kwargs, compile_regional):
    actions = parse_recipe(recipe)
    pipe = None

    try:
        flush()
        pipe = load_pipeline(config)
        notes = apply_recipe(
            pipe,
            actions,
            call_kwargs=config.pipeline_call_kwargs,
            compile_kwargs=compile_kwargs,
            compile_regional=compile_regional,
        )
        mean_ms, std_ms, peak_vram_gb = benchmark_pipeline_call(
            pipe,
            config.pipeline_call_kwargs,
            num_runs=num_runs,
            num_warmups=num_warmups,
        )
        return RecipeBenchmarkResult(
            pipeline=config.name,
            recipe=recipe,
            status="pass",
            mean_ms=round(mean_ms, 1),
            std_ms=round(std_ms, 1),
            peak_vram_gb=round(peak_vram_gb, 2),
            num_runs=num_runs,
            num_warmups=num_warmups,
            notes="; ".join(notes),
        )
    except Exception as error:
        return RecipeBenchmarkResult(
            pipeline=config.name,
            recipe=recipe,
            status="fail",
            mean_ms=None,
            std_ms=None,
            peak_vram_gb=None,
            num_runs=num_runs,
            num_warmups=num_warmups,
            notes=str(error),
        )
    finally:
        if pipe is not None:
            try:
                pipe.to("cpu")
            except Exception:
                pass
            del pipe
        flush()


def write_json_report(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown_report(path: str, payload: dict):
    lines = [
        f"# Pipeline recipe report: {payload['pipeline']}",
        "",
        f"- Generated: {payload['generated_at_utc']}",
        f"- Device: {payload['device_name']}",
        f"- Torch: {payload['torch_version']}",
        f"- Call kwargs: `{json.dumps(payload['call_kwargs'], sort_keys=True)}`",
        "",
        "| Recipe | Status | Mean (ms) | Std (ms) | Peak VRAM (GB) | Notes |",
        "|---|---|---:|---:|---:|---|",
    ]

    for result in payload["results"]:
        lines.append(
            "| {recipe} | {status} | {mean} | {std} | {peak} | {notes} |".format(
                recipe=result["recipe"],
                status=result["status"],
                mean=result["mean_ms"] if result["mean_ms"] is not None else "—",
                std=result["std_ms"] if result["std_ms"] is not None else "—",
                peak=result["peak_vram_gb"] if result["peak_vram_gb"] is not None else "—",
                notes=result["notes"] or "—",
            )
        )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark pipeline optimization recipes")
    parser.add_argument(
        "--pipeline",
        choices=["flux", "flux2", "wan", "ltx2", "qwenimage"],
        required=True,
        help="Which pipeline to benchmark",
    )
    parser.add_argument(
        "--recipe",
        action="append",
        default=None,
        help=(
            "Recipe string to run. Combine actions with '+', for example "
            "'group_offload_leaf+layerwise_casting' or 'compile+attention:_native_flash'. "
            "Defaults to a conservative built-in recipe set."
        ),
    )
    parser.add_argument("--output_dir", default="profiling_results", help="Directory for report output")
    parser.add_argument("--num_steps", type=int, default=None, help="Override num_inference_steps")
    parser.add_argument("--full_decode", action="store_true", help="Set output_type='pil' so VAE recipes are meaningful")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of timed runs")
    parser.add_argument("--num_warmups", type=int, default=2, help="Number of warmup runs")
    parser.add_argument(
        "--compile_mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode for recipes that include compile",
    )
    parser.add_argument("--compile_fullgraph", action="store_true", help="Use fullgraph=True for torch.compile")
    parser.add_argument(
        "--compile_regional",
        action="store_true",
        help="Use compile_repeated_blocks() when available instead of full transformer compilation",
    )
    args = parser.parse_args()

    registry = build_registry()
    config = copy.deepcopy(registry[args.pipeline])

    if args.num_steps is not None:
        config.pipeline_call_kwargs["num_inference_steps"] = args.num_steps
    if args.full_decode:
        config.pipeline_call_kwargs["output_type"] = "pil"

    recipes = args.recipe or list(DEFAULT_RECIPES)
    compile_kwargs = {"fullgraph": args.compile_fullgraph, "mode": args.compile_mode}

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    for recipe in recipes:
        logger.info("Running recipe %s for pipeline %s", recipe, config.name)
        results.append(
            run_recipe(
                config,
                recipe,
                num_runs=args.num_runs,
                num_warmups=args.num_warmups,
                compile_kwargs=compile_kwargs,
                compile_regional=args.compile_regional,
            )
        )

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    payload = {
        "pipeline": config.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "call_kwargs": config.pipeline_call_kwargs,
        "results": [asdict(result) for result in results],
    }

    json_path = os.path.join(args.output_dir, f"{config.name}_recipes.json")
    md_path = os.path.join(args.output_dir, f"{config.name}_recipes.md")
    write_json_report(json_path, payload)
    write_markdown_report(md_path, payload)

    logger.info("Wrote JSON report to %s", json_path)
    logger.info("Wrote Markdown report to %s", md_path)


if __name__ == "__main__":
    main()
