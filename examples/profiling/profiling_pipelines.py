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

import torch
from profiling_utils import PipelineProfiler, PipelineProfilingConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROMPT = "A cat holding a sign that says hello world"


def build_registry():
    """Build the pipeline config registry. Imports are deferred to avoid loading all pipelines upfront."""
    from diffusers import Flux2KleinPipeline, FluxPipeline, LTX2Pipeline, QwenImagePipeline, WanPipeline

    return {
        "flux": PipelineProfilingConfig(
            name="flux",
            pipeline_cls=FluxPipeline,
            pipeline_init_kwargs={
                "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
                "torch_dtype": torch.bfloat16,
            },
            pipeline_call_kwargs={
                "prompt": PROMPT,
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 4,
                "guidance_scale": 3.5,
                "output_type": "latent",
            },
        ),
        "flux2": PipelineProfilingConfig(
            name="flux2",
            pipeline_cls=Flux2KleinPipeline,
            pipeline_init_kwargs={
                "pretrained_model_name_or_path": "black-forest-labs/FLUX.2-klein-base-9B",
                "torch_dtype": torch.bfloat16,
            },
            pipeline_call_kwargs={
                "prompt": PROMPT,
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 4,
                "guidance_scale": 3.5,
                "output_type": "latent",
            },
        ),
        "wan": PipelineProfilingConfig(
            name="wan",
            pipeline_cls=WanPipeline,
            pipeline_init_kwargs={
                "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                "torch_dtype": torch.bfloat16,
            },
            pipeline_call_kwargs={
                "prompt": PROMPT,
                "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                "height": 480,
                "width": 832,
                "num_frames": 81,
                "num_inference_steps": 4,
                "output_type": "latent",
            },
        ),
        "ltx2": PipelineProfilingConfig(
            name="ltx2",
            pipeline_cls=LTX2Pipeline,
            pipeline_init_kwargs={
                "pretrained_model_name_or_path": "Lightricks/LTX-2",
                "torch_dtype": torch.bfloat16,
            },
            pipeline_call_kwargs={
                "prompt": PROMPT,
                "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
                "height": 512,
                "width": 768,
                "num_frames": 121,
                "num_inference_steps": 4,
                "guidance_scale": 4.0,
                "output_type": "latent",
            },
        ),
        "qwenimage": PipelineProfilingConfig(
            name="qwenimage",
            pipeline_cls=QwenImagePipeline,
            pipeline_init_kwargs={
                "pretrained_model_name_or_path": "Qwen/Qwen-Image",
                "torch_dtype": torch.bfloat16,
            },
            pipeline_call_kwargs={
                "prompt": PROMPT,
                "negative_prompt": " ",
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 4,
                "true_cfg_scale": 4.0,
                "output_type": "latent",
            },
        ),
    }


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
