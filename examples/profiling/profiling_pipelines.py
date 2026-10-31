"""
Profile diffusers pipelines with torch.profiler.

Usage:
    python profiling/profiling_pipelines.py --pipeline flux --mode eager
    python profiling/profiling_pipelines.py --pipeline flux --mode compile
    python profiling/profiling_pipelines.py --pipeline flux --mode both
    python profiling/profiling_pipelines.py --pipeline all --mode eager
    python profiling/profiling_pipelines.py --pipeline wan --mode eager --full_decode
    python profiling/profiling_pipelines.py --pipeline flux --mode compile --num_steps 4

Benchmarking (wall-clock time, no profiler overhead):
    python profiling/profiling_pipelines.py --pipeline flux --mode compile --benchmark
    python profiling/profiling_pipelines.py --pipeline flux --mode both --benchmark --num_runs 10 --num_warmups 3
"""

import argparse
import copy
import logging

try:
    from .pipeline_registry import build_registry
    from .profiling_utils import PipelineProfiler
except ImportError:
    from pipeline_registry import build_registry
    from profiling_utils import PipelineProfiler


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Profile diffusers pipelines with torch.profiler")
    parser.add_argument(
        "--pipeline",
        choices=["flux", "flux2", "wan", "ltx2", "qwenimage", "all"],
        required=True,
        help="Which pipeline to profile",
    )
    parser.add_argument(
        "--mode",
        choices=["eager", "compile", "both"],
        default="eager",
        help="Run in eager mode, compile mode, or both",
    )
    parser.add_argument("--output_dir", default="profiling_results", help="Directory for trace output")
    parser.add_argument("--num_steps", type=int, default=None, help="Override num_inference_steps")
    parser.add_argument("--full_decode", action="store_true", help="Profile including VAE decode (output_type='pil')")
    parser.add_argument(
        "--compile_mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode",
    )
    parser.add_argument("--compile_fullgraph", action="store_true", help="Use fullgraph=True for torch.compile")
    parser.add_argument(
        "--compile_regional",
        action="store_true",
        help="Use compile_repeated_blocks() instead of full model compile",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark wall-clock time instead of profiling. Uses CUDA events, no profiler overhead.",
    )
    parser.add_argument("--num_runs", type=int, default=5, help="Number of timed runs for benchmarking")
    parser.add_argument("--num_warmups", type=int, default=2, help="Number of warmup runs for benchmarking")
    args = parser.parse_args()

    registry = build_registry()

    pipeline_names = list(registry.keys()) if args.pipeline == "all" else [args.pipeline]
    modes = ["eager", "compile"] if args.mode == "both" else [args.mode]

    for pipeline_name in pipeline_names:
        for mode in modes:
            config = copy.deepcopy(registry[pipeline_name])

            # Apply overrides
            if args.num_steps is not None:
                config.pipeline_call_kwargs["num_inference_steps"] = args.num_steps
            if args.full_decode:
                config.pipeline_call_kwargs["output_type"] = "pil"
            if mode == "compile":
                config.compile_kwargs = {
                    "fullgraph": args.compile_fullgraph,
                    "mode": args.compile_mode,
                }
                config.compile_regional = args.compile_regional

            profiler = PipelineProfiler(config, args.output_dir)
            try:
                if args.benchmark:
                    logger.info(f"Benchmarking {pipeline_name} in {mode} mode...")
                    profiler.benchmark(num_runs=args.num_runs, num_warmups=args.num_warmups)
                else:
                    logger.info(f"Profiling {pipeline_name} in {mode} mode...")
                    trace_file = profiler.run()
                    logger.info(f"Done: {trace_file}")
            except Exception as e:
                logger.error(f"Failed to {'benchmark' if args.benchmark else 'profile'} {pipeline_name} ({mode}): {e}")


if __name__ == "__main__":
    main()
