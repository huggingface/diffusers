import gc
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd
import torch
import torch.utils.benchmark as benchmark

from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.testing_utils import require_torch_gpu, torch_device


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


def model_init_fn(model_cls, group_offload_kwargs=None, layerwise_upcasting=False, **init_kwargs):
    model = model_cls.from_pretrained(**init_kwargs).eval()
    if group_offload_kwargs and isinstance(group_offload_kwargs, dict):
        model.enable_group_offload(**group_offload_kwargs)
    else:
        model.to(torch_device)
    if layerwise_upcasting:
        model.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=init_kwargs.get("torch_dtype", torch.bfloat16)
        )
    return model


@dataclass
class BenchmarkScenario:
    name: str
    model_cls: ModelMixin
    model_init_kwargs: Dict[str, Any]
    model_init_fn: Callable
    get_model_input_dict: Callable
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
            model_cls=scenario.model_cls,
            init_fn=scenario.model_init_fn,
            init_kwargs=scenario.model_init_kwargs,
            get_input_fn=scenario.get_model_input_dict,
            compile_kwargs=None,
        )

        # 2) compiled stats (if any)
        compiled = {"time": None, "memory": None}
        if scenario.compile_kwargs:
            compiled = self._run_phase(
                model_cls=scenario.model_cls,
                init_fn=scenario.model_init_fn,
                init_kwargs=scenario.model_init_kwargs,
                get_input_fn=scenario.get_model_input_dict,
                compile_kwargs=scenario.compile_kwargs,
            )

        # 3) merge
        result = {
            "scenario": scenario.name,
            "model_cls": scenario.model_cls.__name__,
            "time_plain_s": plain["time"],
            "mem_plain_GB": plain["memory"],
            "time_compile_s": compiled["time"],
            "mem_compile_GB": compiled["memory"],
        }
        if scenario.compile_kwargs:
            result["fullgraph"] = scenario.compile_kwargs.get("fullgraph", False)
            result["mode"] = scenario.compile_kwargs.get("mode", "default")
        else:
            result["fullgraph"], result["mode"] = None, None
        return result

    def run_bencmarks_and_collate(self, scenarios: Union[BenchmarkScenario, list[BenchmarkScenario]], filename: str):
        if not isinstance(scenarios, list):
            scenarios = [scenarios]
        records = [self.run_benchmark(s) for s in scenarios]
        df = pd.DataFrame.from_records(records)
        df.to_csv(filename, index=False)

    def _run_phase(
        self,
        *,
        model_cls: ModelMixin,
        init_fn: Callable,
        init_kwargs: Dict[str, Any],
        get_input_fn: Callable,
        compile_kwargs: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        # setup
        self.pre_benchmark()

        # init & (optional) compile
        model = init_fn(model_cls, **init_kwargs)
        if compile_kwargs:
            model.compile(**compile_kwargs)

        # build inputs
        inp = get_input_fn()

        # measure
        run_ctx = torch._inductor.utils.fresh_inductor_cache() if compile_kwargs else nullcontext()
        with run_ctx:
            time_s = benchmark_fn(lambda m, d: m(**d), model, inp)
        mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        mem_gb = round(mem_gb, 2)

        # teardown
        self.post_benchmark(model)
        del model
        return {"time": time_s, "memory": mem_gb}
