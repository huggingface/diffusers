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

"""Generic torchrun worker: assert a model's Neuron tensor-parallel output matches its single-device reference.

Model-agnostic. The model under test is supplied as a ``module:function`` spec reference on the command line; the
referenced factory returns ``(model_class, init_dict, inputs)`` with CPU tensors, so all model-specific test data lives
with the launching test rather than here.

Launched as a subprocess by a ``@require_torch_neuron`` test (and runnable directly for debugging)::

    torchrun --nproc_per_node=2 _neuron_tp_worker.py \\
        tests.models.transformers.test_models_transformer_flux2:make_neuron_tp_spec

Each rank builds an identical (seeded) model on CPU, computes a single-device reference, then shards it with
``enable_parallelism(TensorParallelConfig(mesh=neuron_mesh))`` — which auto-selects the Neuron pre-shard backend — runs
a forward pass on the Neuron device, and asserts the gathered output matches the reference. Exit code 0 means the TP
path is numerically equivalent to the unsharded model; non-zero means failure.
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
from torch.distributed.device_mesh import DeviceMesh

from diffusers import TensorParallelConfig


def main():
    parser = argparse.ArgumentParser(description="Neuron tensor-parallel correctness worker.")
    parser.add_argument(
        "spec",
        help="`module:function` reference returning (model_class, init_dict, cpu_inputs) for the model under test.",
    )
    args = parser.parse_args()
    module_name, _, fn_name = args.spec.partition(":")
    model_class, init_dict, inputs = getattr(importlib.import_module(module_name), fn_name)()

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    tp_size = dist.get_world_size()
    device = torch.neuron.current_device()
    tp_mesh = DeviceMesh("neuron", list(range(tp_size)))

    # Identical weights on every rank (same seed), kept on CPU as the Neuron pre-shard backend requires.
    torch.manual_seed(0)
    model = model_class(**init_dict).eval()

    # Single-device (unsharded) reference on CPU, computed before TP mutates the weights in place.
    with torch.no_grad():
        ref_output = model(**inputs, return_dict=False)[0].float().cpu()

    # Shard across all ranks; the Neuron backend is auto-selected from the mesh device type.
    model.enable_parallelism(config=TensorParallelConfig(mesh=tp_mesh))
    model = model.to(device)
    torch.neuron.synchronize()

    inputs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        tp_output = model(**inputs_on_device, return_dict=False)[0]
    torch.neuron.synchronize()
    tp_output = tp_output.float().cpu()

    if rank == 0:
        assert tp_output.shape == ref_output.shape, f"shape mismatch: {tp_output.shape} vs {ref_output.shape}"
        assert torch.isfinite(tp_output).all(), "TP output contains non-finite values"
        max_abs = (tp_output - ref_output).abs().max().item()
        denom = ref_output.abs().max().item() + 1e-6
        print(
            f"[rank0] tp_size={tp_size} output_shape={tuple(tp_output.shape)} "
            f"max_abs_diff={max_abs:.4e} max_rel_diff={max_abs / denom:.4e}"
        )
        # Neuron runs matmuls in bf16 internally, so compare with a bf16-level tolerance. A wrong shard
        # plan produces grossly different output and is caught comfortably within this bound.
        torch.testing.assert_close(tp_output, ref_output, atol=2e-2, rtol=2e-2)
        print("[rank0] PASS: Neuron tensor-parallel output matches single-device reference.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        # Ensure a non-zero exit so the launching pytest sees the failure.
        os._exit(1)
