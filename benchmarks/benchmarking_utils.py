import gc
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.utils.benchmark as benchmark

from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.testing_utils import require_torch_gpu


def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=1,
    )
    return float(f"{(t0.blocked_autorange().mean):.3f}")


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


@dataclass
class BenchmarkScenario:
    name: str
    model_cls: ModelMixin
    model_init_kwargs: Dict[str, Any]
    model_init_fn: Callable
    get_model_input_dict: Callable[[], Dict[str, Any]]
    compile_kwargs: Optional[Dict[str, Any]] = None


@require_torch_gpu
class BenchmarkMixin:
    def pre_benchmark(self):
        flush()
        torch.compiler.reset()

    def post_benchmark(self, model):
        model.cpu()
        flush()
        torch.compiler.reset()

    @torch.no_grad()
    def run_benchmark(self, scenario: BenchmarkScenario):
        # 1) plain stats
        plain = self._run_phase(
            init_fn=scenario.model_init_fn,
            init_kwargs=scenario.model_init_kwargs,
            get_input_fn=scenario.get_model_input_dict,
            compile_kwargs=None,
        )

        # 2) compiled stats (if any)
        compiled = None
        if scenario.compile_kwargs:
            compiled = self._run_phase(
                init_fn=scenario.model_init_fn,
                init_kwargs=scenario.model_init_kwargs,
                get_input_fn=scenario.get_model_input_dict,
                compile_kwargs=scenario.compile_kwargs,
            )

        # 3) merge
        result = {"scenario": scenario.name, "time_plain_s": plain["time"], "mem_plain_GB": plain["memory"]}
        if compiled:
            result.update(
                {
                    "time_compile_s": compiled["time"],
                    "mem_compile_GB": compiled["memory"],
                }
            )
        return result

    def _run_phase(
        self,
        *,
        init_fn: Callable[..., Any],
        init_kwargs: Dict[str, Any],
        get_input_fn: Callable[[], Dict[str, torch.Tensor]],
        compile_kwargs: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        # setup
        self.pre_benchmark()

        # init & (optional) compile
        model = init_fn(**init_kwargs)
        if compile_kwargs:
            model.compile(**compile_kwargs)

        # build inputs
        inp = get_input_fn()

        # measure
        time_s = benchmark_fn(lambda m, d: m(**d), model, inp)
        mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        mem_gb = round(mem_gb, 2)

        # teardown
        self.post_benchmark(model)
        del model
        return {"time": time_s, "memory": mem_gb}
