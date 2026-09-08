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

"""Shared `pytest`-side launcher for the per-backend TP-correctness `torchrun` workers.

Used by `TestFlux2TransformerTensorParallelTPU`/`TestFlux2TransformerTensorParallelNeuron` (and any future
accelerator's TP test) to launch their `_<backend>_tp_worker.py` under `torchrun` and assert it exits cleanly.
"""

import os
import subprocess
import sys

import pytest


def run_tp_worker_subprocess(worker_filename: str, spec: str, world_size: int, timeout_s: int = 900) -> None:
    """Launch a `torchrun` TP-correctness worker subprocess and assert it exits cleanly.

    Args:
        worker_filename: Name of the worker script, resolved relative to this file's directory (e.g.
            `"_tpu_tp_worker.py"`).
        spec: `module:function` reference forwarded to the worker, see `_tp_worker_common.run_tp_correctness_worker`.
        world_size: Number of ranks to launch (`torchrun --nproc_per_node`).
        timeout_s: Seconds to wait for the subprocess before failing the test. The worker itself only needs a couple
            of minutes even from a cold compile; this generously bounds it so a real hang (e.g. a distributed-runtime
            barrier timeout) fails the test loudly instead of stalling the run.
    """
    worker = os.path.join(os.path.dirname(__file__), worker_filename)
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={world_size}", worker, spec]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            f"TP worker did not finish within {timeout_s}s (likely stuck on a distributed-runtime barrier).\n"
            f"--- stdout ---\n{e.stdout}\n--- stderr ---\n{e.stderr}"
        ) from e
    assert result.returncode == 0, (
        f"TP worker failed (exit {result.returncode}).\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


class TensorParallelTPUTesterMixin:
    """Mixin for a `@require_torch_tpu` tensor-parallel correctness test, run via `_tpu_tp_worker.py`.

    Subclasses set `TP_SPEC` to a `module:function` reference (see
    `_tp_worker_common.run_tp_correctness_worker`'s `spec` argument) and, only if the model spec's head count
    doesn't divide 4, override `WORLD_SIZE`.

    `WORLD_SIZE` defaults to 4 rather than an arbitrary rank count: `torch_tpu`'s per-generation topology table
    (`torch_tpu._internal.utils.hardware`) only enumerates whole-pod-slice chip counts (1/4/8 for v6e, for example),
    not arbitrary sub-slices of a larger single host. A rank count with no matching whole-slice topology has
    nothing to advertise and the PJRT client never completes its start-session barrier — the test would hang for
    the barrier's full multi-minute timeout instead of failing. 4 is the smallest whole-slice count every current
    TPU generation defines (see `_V4_TOPOLOGY` / `_V5E_TOPOLOGY` / `_V6E_TOPOLOGY` / `_V7_TOPOLOGY` in
    `torch_tpu._internal.utils.hardware`). `skip_if_unsupported` below still checks the actual host up front and
    skips fast instead of hanging when it doesn't have exactly that many chips.

    Requires `TORCH_TPU_TOPOLOGY` and `TORCH_TPU_SLICEBUILDER_ADDRESSES` to be set. Source them via::

        eval $(python -m torch_tpu._internal.distributed.launchers.singlehost_wrapper | sed 's/^/export /')
    """

    WORLD_SIZE = 4
    # The worker itself only needs a couple of minutes even from a cold XLA compile; this generously bounds the
    # subprocess so a real hang (e.g. a barrier timeout this skip failed to catch) fails the test loudly instead of
    # stalling the run.
    TIMEOUT_S = 900
    TP_SPEC: str = ""

    def skip_if_unsupported(self):
        """Skip unless the host has exactly `WORLD_SIZE` TPU chips.

        A topology *string* existing for a chip count (`hardware.get_tpu_topology`) isn't enough to guarantee the
        PJRT client can actually form that session: a sub-slice of a larger single host (e.g. claiming 2 of a
        4-chip v6e-4's chips via `TORCH_TPU_TOPOLOGY`/`TORCH_TPU_SLICEBUILDER_ADDRESSES`) can still fail with a
        low-level `START_SESSION` GRPC error, since the runtime's session setup is tied to the host's actual
        provisioned slice, not just a topology label. The only combination verified to work is running with exactly
        as many ranks as the host has chips.
        """
        from torch_tpu._internal.utils import hardware

        try:
            device_count = hardware.get_tpu_device_count()
        except Exception as e:  # pragma: no cover - defensive, hardware detection is best-effort
            pytest.skip(f"Could not determine local TPU chip count: {e}")
            return

        if device_count != self.WORLD_SIZE:
            pytest.skip(
                f"This host exposes {device_count} TPU chip(s), but this test requires exactly "
                f"{self.WORLD_SIZE} (a TPU single-host tensor-parallel job must use all chips on the host; "
                f"sub-slicing a larger host is not reliably supported by the runtime). Run this test on a host "
                f"with exactly {self.WORLD_SIZE} TPU chips."
            )

    def test_tensor_parallel_tpu_inference(self):
        self.skip_if_unsupported()
        run_tp_worker_subprocess(
            "_tpu_tp_worker.py", self.TP_SPEC, world_size=self.WORLD_SIZE, timeout_s=self.TIMEOUT_S
        )
