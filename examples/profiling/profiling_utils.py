import functools
import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.profiler


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def annotate(func, name):
    """Wrap a function with torch.profiler.record_function for trace annotation."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.profiler.record_function(name):
            return func(*args, **kwargs)

    return wrapper


def annotate_pipeline(pipe):
    """Apply profiler annotations to key pipeline methods.

    Monkey-patches bound methods so they appear as named spans in the trace.
    Non-invasive — no source modifications required.
    """
    annotations = [
        ("transformer", "forward", "transformer_forward"),
        ("vae", "decode", "vae_decode"),
        ("vae", "encode", "vae_encode"),
        ("scheduler", "step", "scheduler_step"),
    ]

    # Annotate sub-component methods
    for component_name, method_name, label in annotations:
        component = getattr(pipe, component_name, None)
        if component is None:
            continue
        method = getattr(component, method_name, None)
        if method is None:
            continue
        setattr(component, method_name, annotate(method, label))

    # Annotate pipeline-level methods
    if hasattr(pipe, "encode_prompt"):
        pipe.encode_prompt = annotate(pipe.encode_prompt, "encode_prompt")


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


@dataclass
class PipelineProfilingConfig:
    name: str
    pipeline_cls: Any
    pipeline_init_kwargs: dict[str, Any]
    pipeline_call_kwargs: dict[str, Any]
    compile_kwargs: dict[str, Any] | None = field(default=None)
    compile_regional: bool = False


class PipelineProfiler:
    def __init__(self, config: PipelineProfilingConfig, output_dir: str = "profiling_results"):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def setup_pipeline(self):
        """Load the pipeline from pretrained, optionally compile, and annotate."""
        logger.info(f"Loading pipeline: {self.config.name}")
        pipe = self.config.pipeline_cls.from_pretrained(**self.config.pipeline_init_kwargs)
        pipe.to("cuda")

        if self.config.compile_kwargs:
            if self.config.compile_regional:
                logger.info(
                    f"Regional compilation (compile_repeated_blocks) with kwargs: {self.config.compile_kwargs}"
                )
                pipe.transformer.compile_repeated_blocks(**self.config.compile_kwargs)
            else:
                logger.info(f"Full compilation with kwargs: {self.config.compile_kwargs}")
                pipe.transformer.compile(**self.config.compile_kwargs)

        # Disable tqdm progress bar to avoid CPU overhead / IO between steps
        pipe.set_progress_bar_config(disable=True)

        annotate_pipeline(pipe)
        return pipe

    def run(self):
        """Execute the profiling run: warmup, then profile one pipeline call."""
        pipe = self.setup_pipeline()
        flush()

        mode = "compile" if self.config.compile_kwargs else "eager"
        trace_file = os.path.join(self.output_dir, f"{self.config.name}_{mode}.json")

        # Warmup (pipeline __call__ is already decorated with @torch.no_grad())
        logger.info("Running warmup...")
        pipe(**self.config.pipeline_call_kwargs)
        flush()

        # Profile
        logger.info("Running profiled iteration...")
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with torch.profiler.record_function("pipeline_call"):
                pipe(**self.config.pipeline_call_kwargs)

        # Export trace
        prof.export_chrome_trace(trace_file)
        logger.info(f"Chrome trace saved to: {trace_file}")

        # Print summary
        print("\n" + "=" * 80)
        print(f"Profile summary: {self.config.name} ({mode})")
        print("=" * 80)
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=20,
            )
        )

        # Cleanup
        pipe.to("cpu")
        del pipe
        flush()

        return trace_file
