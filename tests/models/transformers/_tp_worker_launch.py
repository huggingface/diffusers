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
