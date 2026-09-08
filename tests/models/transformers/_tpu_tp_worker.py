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

"""TPU entry point for the generic TP-correctness worker (see `_tp_worker_common.py`).

Model-agnostic. The model under test is supplied as a ``module:function`` spec reference on the command line; the
referenced factory returns ``(model_class, init_dict, inputs)`` with CPU tensors, so all model-specific test data lives
with the launching test rather than here.

Launched as a subprocess by a ``@require_torch_tpu`` test (and runnable directly for debugging)::

    eval $(python -m torch_tpu._internal.distributed.launchers.singlehost_wrapper | sed 's/^/export /')
    torchrun --nproc_per_node=4 _tpu_tp_worker.py \\
        tests.models.transformers.test_models_transformer_flux2:make_tpu_tp_spec

Exit code 0 means the TP path is numerically equivalent to the unsharded model; non-zero means failure.
"""

import argparse
import os
import sys
import traceback


# Make the in-repo `diffusers` and `tests` packages importable when run via torchrun from an arbitrary CWD.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import torch.distributed as dist
import torch_tpu  # noqa: F401 — registers "tpu" device and "tpu_dist" backend
from torch_tpu._internal import sync as tpu_sync

from tests.models.transformers._tp_worker_common import run_tp_correctness_worker


def main():
    parser = argparse.ArgumentParser(description="TPU tensor-parallel correctness worker.")
    parser.add_argument(
        "spec",
        help="`module:function` reference returning (model_class, init_dict, tpu_inputs) for the model under test.",
    )
    args = parser.parse_args()

    dist.init_process_group(backend="tpu_dist")
    # The reference runs on the TPU (not CPU) so both the reference and the TP pass use the same Flash Attention
    # kernel; the only difference between them is sharding, not numerical implementation. TPU Flash Attention has
    # bf16-level numerics, so the tolerance is wider than fp32 — but a wrong shard plan produces grossly different
    # output and is caught comfortably within this bound.
    run_tp_correctness_worker(
        args.spec,
        mesh_device_type="tpu",
        to_device="tpu",
        backend_label="TPU",
        synchronize=lambda: tpu_sync.synchronize(None, wait=True),
        reference_on_device=True,
        atol=0.1,
        rtol=0.1,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        # Ensure a non-zero exit so the launching pytest sees the failure.
        os._exit(1)
