import gc
import inspect
import logging
import os
import queue
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd
import torch
import torch.utils.benchmark as benchmark

from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.testing_utils import require_torch_gpu, torch_device


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

NUM_WARMUP_ROUNDS = 5


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


# Adapted from https://github.com/lucasb-eyer/cnn_vit_benchmarks/blob/15b665ff758e8062131353076153905cae00a71f/main.py
def calculate_flops(model, input_dict):
    try:
        from torchprofile import profile_macs
    except ModuleNotFoundError:
        raise

    # This is a hacky way to convert the kwargs to args as `profile_macs` cries about kwargs.
    sig = inspect.signature(model.forward)
    param_names = [
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and p.name != "self"
    ]
    bound = sig.bind_partial(**input_dict)
    bound.apply_defaults()
    args = tuple(bound.arguments[name] for name in param_names)

    model.eval()
    with torch.no_grad():
        macs = profile_macs(model, args)
    flops = 2 * macs  # 1 MAC operation = 2 FLOPs (1 multiplication + 1 addition)
    return flops


def calculate_params(model):
    return sum(p.numel() for p in model.parameters())


# Users can define their own in case this doesn't suffice. For most cases,
# it should be sufficient.
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
        # 0) Basic stats
        logger.info(f"Running scenario: {scenario.name}.")
        try:
            model = model_init_fn(scenario.model_cls, **scenario.model_init_kwargs)
            num_params = round(calculate_params(model) / 1e9, 2)
            try:
                flops = round(calculate_flops(model, input_dict=scenario.get_model_input_dict()) / 1e9, 2)
            except Exception as e:
                logger.info(f"Problem in calculating FLOPs:\n{e}")
                flops = None
            model.cpu()
            del model
        except Exception as e:
            logger.info(f"Error while initializing the model and calculating FLOPs:\n{e}")
            return {}
        self.pre_benchmark()

        # 1) plain stats
        results = {}
        plain = None
        try:
            plain = self._run_phase(
                model_cls=scenario.model_cls,
                init_fn=scenario.model_init_fn,
                init_kwargs=scenario.model_init_kwargs,
                get_input_fn=scenario.get_model_input_dict,
                compile_kwargs=None,
            )
        except Exception as e:
            logger.info(f"Benchmark could not be run with the following error:\n{e}")
            return results

        # 2) compiled stats (if any)
        compiled = {"time": None, "memory": None}
        if scenario.compile_kwargs:
            try:
                compiled = self._run_phase(
                    model_cls=scenario.model_cls,
                    init_fn=scenario.model_init_fn,
                    init_kwargs=scenario.model_init_kwargs,
                    get_input_fn=scenario.get_model_input_dict,
                    compile_kwargs=scenario.compile_kwargs,
                )
            except Exception as e:
                logger.info(f"Compilation benchmark could not be run with the following error\n: {e}")
                if plain is None:
                    return results

        # 3) merge
        result = {
            "scenario": scenario.name,
            "model_cls": scenario.model_cls.__name__,
            "num_params_B": num_params,
            "flops_G": flops,
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
        record_queue = queue.Queue()
        stop_signal = object()

        def _writer_thread():
            while True:
                item = record_queue.get()
                if item is stop_signal:
                    break
                df_row = pd.DataFrame([item])
                write_header = not os.path.exists(filename)
                df_row.to_csv(filename, mode="a", header=write_header, index=False)
                record_queue.task_done()

            record_queue.task_done()

        writer = threading.Thread(target=_writer_thread, daemon=True)
        writer.start()

        for s in scenarios:
            try:
                record = self.run_benchmark(s)
                if record:
                    record_queue.put(record)
                else:
                    logger.info(f"Record empty from scenario: {s.name}.")
            except Exception as e:
                logger.info(f"Running scenario ({s.name}) led to error:\n{e}")
        record_queue.put(stop_signal)
        logger.info(f"Results serialized to {filename=}.")

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
            for _ in range(NUM_WARMUP_ROUNDS):
                _ = model(**inp)
            time_s = benchmark_fn(lambda m, d: m(**d), model, inp)
        mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        mem_gb = round(mem_gb, 2)

        # teardown
        self.post_benchmark(model)
        del model
        return {"time": time_s, "memory": mem_gb}
