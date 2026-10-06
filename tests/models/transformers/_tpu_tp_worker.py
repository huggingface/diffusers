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

"""`torchrun` worker: check a model's TPU tensor-parallel output against its single-chip output.

torchrun --nproc_per_node=4 _tpu_tp_worker.py tests.models.transformers.test_models_transformer_flux2:make_tpu_tp_spec
"""

import argparse
import copy
import importlib
import os
import sys
import traceback


# Import the in-repo `diffusers` and `tests`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import torch
import torch.distributed as dist
import torch_tpu  # noqa: F401 — registers "tpu" device and "tpu_dist" backend
from torch.distributed.device_mesh import DeviceMesh

from diffusers import TensorParallelConfig


def main():
    parser = argparse.ArgumentParser(description="TPU tensor-parallel correctness worker.")
    parser.add_argument(
        "spec",
        help="`module:function` reference returning (model_class, init_dict, cpu_inputs) for the model under test.",
    )
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()
    module_name, _, fn_name = args.spec.partition(":")
    model_class, init_dict, inputs = getattr(importlib.import_module(module_name), fn_name)()

    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    tp_size = dist.get_world_size()

    # TPU runs fp32 matmuls in bf16 by default; use full precision to keep the tolerance tight.
    torch.set_float32_matmul_precision("highest")

    # Identical weights on every rank (same seed).
    torch.manual_seed(0)
    model = model_class(**init_dict).eval()

    # Unsharded reference on TPU, so only the sharding differs.
    ref_model = copy.deepcopy(model).to("tpu")
    inputs_on_device = {k: v.to("tpu") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        ref_output = ref_model(**inputs_on_device, return_dict=False)[0]
    ref_output = ref_output.float().cpu()
    del ref_model

    model.enable_parallelism(config=TensorParallelConfig(mesh=DeviceMesh("tpu", list(range(tp_size)))))
    model = model.to("tpu")
    with torch.no_grad():
        tp_output = model(**inputs_on_device, return_dict=False)[0]
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
        torch.testing.assert_close(tp_output, ref_output, atol=args.atol, rtol=args.rtol)
        print("[rank0] PASS: TPU tensor-parallel output matches single-device reference.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        # Non-zero exit so pytest sees the failure.
        os._exit(1)
