import gc

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
    return f"{(t0.blocked_autorange().mean):.3f}"


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


@require_torch_gpu
class BenchmarkMixin:
    model_class: ModelMixin = None
    compile_kwargs: dict = None

    def get_model_init_dict(self):
        raise NotImplementedError

    def initialize_model(self):
        raise NotImplementedError

    def get_input_dict(self):
        raise NotImplementedError

    def pre_benchmark(self):
        flush()
        torch.compiler.reset()

    def post_benchmark(self, model):
        model.cpu()
        flush()
        torch.compiler.reset()

    @torch.no_grad()
    def run_benchmark(self):
        self.pre_benchmark()

        model = self.initialize_model()  # Takes care of device placement.
        input_dict = self.get_input_dict()  # Takes care of device placement.

        time = benchmark_fn(lambda model, input_dict: model(**input_dict), model, input_dict)
        memory = torch.cuda.max_memory_allocated() / (1024**3)
        memory = float(f"{memory:.2f}")
        non_compile_stats = {"time": time, "memory": memory}

        self.post_benchmark(model)
        del model
        self.pre_benchmark()

        compile_stats = None
        if self.compile_kwargs is not None:
            model = self.initialize_model()
            input_dict = self.get_input_dict()
            model.compile(**self.compile_kwargs)
            time = benchmark_fn(lambda model, input_dict: model(**input_dict), model, input_dict)
            memory = torch.cuda.max_memory_allocated() / (1024**3)
            memory = float(f"{memory:.2f}")
            compile_stats = {"time": time, "memory": memory}

        self.post_benchmark(model)
        del model
        return non_compile_stats, compile_stats
