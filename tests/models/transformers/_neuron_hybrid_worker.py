# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

"""Generic torchrun worker: assert a model's Neuron tensor-parallel x context-parallel output matches its reference.

The counterpart of `_neuron_tp_worker.py` for the two parallelisms composed in a single `ParallelConfig`. Same
contract: the model under test is supplied as a `module:function` spec reference on the command line, and the
referenced factory returns `(model_class, init_dict, inputs)` with CPU tensors.

    torchrun --nproc_per_node=8 _neuron_hybrid_worker.py \\
        tests.models.transformers.test_models_transformer_flux:make_neuron_hybrid_spec

`tp_degree` and `ulysses_degree` are read from `TP_DEGREE` / `ULYSSES_DEGREE` (defaults 2 and 4, whose product is
the launched world size). `ulysses_degree` cannot be 2 on Neuron: its all-to-all only accepts group sizes of 4, 8,
16 or multiples of 32.

No mesh is passed, so this also exercises the default mesh layout that `enable_parallelism` builds for the combined
case -- which matters on Neuron, where the all-to-all Ulysses depends on rejects strided replica groups and so the
context-parallel dimensions have to vary fastest.

Exit code 0 means the composed path is numerically equivalent to the unsharded model; non-zero means failure.
"""

import argparse
import importlib
import os
import sys
import traceback


# Make the in-repo `diffusers` and `tests` packages importable when run via torchrun from an arbitrary CWD.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import torch
import torch.distributed as dist
import torch_neuronx  # noqa: F401 — registers torch.neuron

from diffusers import ContextParallelConfig, ParallelConfig, TensorParallelConfig


def main():
    parser = argparse.ArgumentParser(description="Neuron tensor-parallel x context-parallel correctness worker.")
    parser.add_argument(
        "spec",
        help="`module:function` reference returning (model_class, init_dict, cpu_inputs) for the model under test.",
    )
    args = parser.parse_args()
    module_name, _, fn_name = args.spec.partition(":")
    model_class, init_dict, inputs = getattr(importlib.import_module(module_name), fn_name)()

    tp_degree = int(os.environ.get("TP_DEGREE", "2"))
    ulysses_degree = int(os.environ.get("ULYSSES_DEGREE", "4"))

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.neuron.current_device()

    if tp_degree * ulysses_degree != world_size:
        raise ValueError(
            f"tp_degree ({tp_degree}) x ulysses_degree ({ulysses_degree}) must equal the world size ({world_size})."
        )

    # Identical weights on every rank (same seed), kept on CPU as the Neuron pre-shard backend requires.
    torch.manual_seed(0)
    model = model_class(**init_dict).eval()

    # Single-device (unsharded) reference on CPU, computed before the shard plan mutates the weights in place.
    with torch.no_grad():
        ref_output = model(**inputs, return_dict=False)[0].float().cpu()

    model.enable_parallelism(
        config=ParallelConfig(
            tensor_parallel_config=TensorParallelConfig(tp_degree=tp_degree),
            context_parallel_config=ContextParallelConfig(ulysses_degree=ulysses_degree),
        )
    )
    model = model.to(device)
    torch.neuron.synchronize()

    inputs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs_on_device, return_dict=False)[0]
    torch.neuron.synchronize()
    output = output.float().cpu()

    if rank == 0:
        assert output.shape == ref_output.shape, f"shape mismatch: {output.shape} vs {ref_output.shape}"
        assert torch.isfinite(output).all(), "output contains non-finite values"
        max_abs = (output - ref_output).abs().max().item()
        denom = ref_output.abs().max().item() + 1e-6
        print(
            f"[rank0] tp_degree={tp_degree} ulysses_degree={ulysses_degree} "
            f"output_shape={tuple(output.shape)} max_abs_diff={max_abs:.4e} max_rel_diff={max_abs / denom:.4e}"
        )
        # Neuron runs matmuls in bf16 internally, so compare with a bf16-level tolerance, as `_neuron_tp_worker`
        # does. A wrong shard plan or a mis-ordered mesh produces grossly different output and is caught well
        # inside this bound.
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=2e-2)
        print("[rank0] PASS: Neuron hybrid-parallel output matches single-device reference.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        # Ensure a non-zero exit so the launching pytest sees the failure.
        os._exit(1)
