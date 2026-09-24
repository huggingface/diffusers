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

"""Shared logic for the per-backend TP-correctness `torchrun` workers (`_tpu_tp_worker.py`, `_neuron_tp_worker.py`).

Both workers follow the same recipe — build an identical model on every rank, compute a single-device reference,
shard with `enable_parallelism`, run on-device, and compare — differing only in backend-specific details (device
string, sync call, whether the reference itself needs to run on-device, and numerical tolerance). This module holds
that shared recipe; each `_<backend>_tp_worker.py` is a thin wrapper supplying those details.
"""

import copy
import importlib
from typing import Callable

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from diffusers import TensorParallelConfig


def run_tp_correctness_worker(
    spec: str,
    *,
    mesh_device_type: str,
    to_device,
    backend_label: str,
    synchronize: Callable[[], None],
    reference_on_device: bool,
    atol: float,
    rtol: float,
) -> None:
    """Assert a model's tensor-parallel output matches its single-device reference, on the given backend.

    Args:
        spec: `module:function` reference returning `(model_class, init_dict, cpu_inputs)` for the model under test.
        mesh_device_type: The `DeviceMesh` device type (e.g. `"tpu"`, `"neuron"`).
        to_device: The value passed to `.to(...)` to move the model/inputs onto the accelerator. Usually the same as
            `mesh_device_type`, but some backends (e.g. Neuron) need a more specific device handle here.
        backend_label: Human-readable backend name for log messages (e.g. `"TPU"`, `"Neuron"`).
        synchronize: Callable that blocks until pending device work completes.
        reference_on_device: If `True`, the unsharded reference forward pass also runs on the accelerator (before TP
            mutates the weights in place), so it uses the same kernels as the TP forward pass and only sharding
            differs. If `False`, the reference runs on CPU.
        atol: Absolute tolerance for the final `torch.testing.assert_close` comparison.
        rtol: Relative tolerance for the final `torch.testing.assert_close` comparison.
    """
    module_name, _, fn_name = spec.partition(":")
    model_class, init_dict, inputs = getattr(importlib.import_module(module_name), fn_name)()

    rank = dist.get_rank()
    tp_size = dist.get_world_size()
    tp_mesh = DeviceMesh(mesh_device_type, list(range(tp_size)))

    # Identical weights on every rank (same seed), kept on CPU as the pre-shard backends require.
    torch.manual_seed(0)
    model = model_class(**init_dict).eval()

    if reference_on_device:
        # Single-device (unsharded) reference on the accelerator, computed before TP mutates the weights in place.
        ref_model = copy.deepcopy(model).to(to_device)
        synchronize()
        inputs_on_device = {k: v.to(to_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            ref_output = ref_model(**inputs_on_device, return_dict=False)[0]
        synchronize()
        ref_output = ref_output.float().cpu()
        del ref_model
    else:
        with torch.no_grad():
            ref_output = model(**inputs, return_dict=False)[0].float().cpu()
        inputs_on_device = {k: v.to(to_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Shard across all ranks; the backend is auto-selected from the mesh device type.
    model.enable_parallelism(config=TensorParallelConfig(mesh=tp_mesh))
    model = model.to(to_device)
    synchronize()

    with torch.no_grad():
        tp_output = model(**inputs_on_device, return_dict=False)[0]
    synchronize()
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
        torch.testing.assert_close(tp_output, ref_output, atol=atol, rtol=rtol)
        print(f"[rank0] PASS: {backend_label} tensor-parallel output matches single-device reference.")

    dist.barrier()
