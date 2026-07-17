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

"""Generic torchrun worker: assert a model's TPU tensor-parallel output matches its single-device reference.

Model-agnostic. The model under test is supplied as a ``module:function`` spec reference on the command line; the
referenced factory returns ``(model_class, init_dict, inputs)`` with CPU tensors, so all model-specific test data lives
with the launching test rather than here.

Launched as a subprocess by a ``@require_torch_tpu`` test (and runnable directly for debugging)::

    eval $(python -m torch_tpu._internal.distributed.launchers.singlehost_wrapper | sed 's/^/export /')
    torchrun --nproc_per_node=2 _tpu_tp_worker.py \\
        tests.models.transformers.test_models_transformer_flux2:make_tpu_tp_spec

Each rank builds an identical (seeded) model on CPU, moves it to the TPU device and computes a single-device reference,
then shards it with ``enable_parallelism(TensorParallelConfig(mesh=tpu_mesh))`` — which auto-selects the TPU pre-shard
backend — runs a forward pass on the TPU device, and asserts the gathered output matches the reference. The reference
runs on the TPU (not CPU) so both the reference and the TP pass use the same Flash Attention kernel; the only
difference between them is sharding, not numerical implementation.

Exit code 0 means the TP path is numerically equivalent to the unsharded model; non-zero means failure.
"""

import argparse
import copy
import importlib
import os
import sys
import traceback


# Make the in-repo `diffusers` and `tests` packages importable when run via torchrun from an arbitrary CWD.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401 — registers "tpu" device and "tpu_dist" backend
from torch.distributed.device_mesh import DeviceMesh
from torch_tpu._internal import sync as tpu_sync

from diffusers import TensorParallelConfig


def main():
    parser = argparse.ArgumentParser(description="TPU tensor-parallel correctness worker.")
    parser.add_argument(
        "spec",
        help="`module:function` reference returning (model_class, init_dict, tpu_inputs) for the model under test.",
    )
    args = parser.parse_args()
    module_name, _, fn_name = args.spec.partition(":")
    model_class, init_dict, inputs = getattr(importlib.import_module(module_name), fn_name)()

    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    tp_size = dist.get_world_size()
    tp_mesh = DeviceMesh("tpu", list(range(tp_size)))

    # Identical weights on every rank (same seed), kept on CPU as the TPU pre-shard backend requires.
    torch.manual_seed(0)
    model = model_class(**init_dict).eval()

    # Single-device (unsharded) reference on the TPU device, computed before TP mutates the weights in place.
    # Using the TPU (not CPU) as the reference device ensures both the reference and the TP forward use the
    # same kernels (e.g. Flash Attention on TPU), so only sharding — not implementation — differs.
    ref_model = copy.deepcopy(model).to("tpu")
    tpu_sync.synchronize(None, wait=True)
    inputs_on_device = {k: v.to("tpu") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        ref_output = ref_model(**inputs_on_device, return_dict=False)[0]
    tpu_sync.synchronize(None, wait=True)
    ref_output = ref_output.float().cpu()
    del ref_model

    # Shard across all ranks; the TPU backend is auto-selected from the mesh device type.
    model.enable_parallelism(config=TensorParallelConfig(mesh=tp_mesh))
    model = model.to("tpu")
    tpu_sync.synchronize(None, wait=True)

    with torch.no_grad():
        tp_output = model(**inputs_on_device, return_dict=False)[0]
    tpu_sync.synchronize(None, wait=True)
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
        # TPU Flash Attention has bf16-level numerics; the tolerance is wider than fp32 but a wrong
        # shard plan produces grossly different output and is caught comfortably within this bound.
        torch.testing.assert_close(tp_output, ref_output, atol=0.1, rtol=0.1)
        print("[rank0] PASS: TPU tensor-parallel output matches single-device reference.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        # Ensure a non-zero exit so the launching pytest sees the failure.
        os._exit(1)
