# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""`diffusers-cli serve --context-parallel` — one worker process per GPU, all running every generation together.

The server process never touches CUDA. It spawns the workers, sends each generation to all of them over a pipe, and
reads the progress and result back from rank 0. Idle workers wait on their pipe rather than inside a collective, so
the process group cannot time out between requests.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import socket
import threading
from argparse import Namespace
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any

from ...utils import logging
from .backends import DiffusersBackend, load_request_media
from .generations import Generation
from .manifest import GMSError, ServedModel


logger = logging.get_logger("diffusers-cli/serve")

_POLL_SECONDS = 0.1
_SHUTDOWN_SECONDS = 30.0


class _WorkerGeneration:
    """The parts of a generation `DiffusersBackend.run` reads, inside a worker.

    `cancel_requested` is a collective: rank 0 reads the flag the server sets and broadcasts it, so every rank
    interrupts at the same step. A rank that stopped alone would leave the others waiting in the next collective.
    """

    def __init__(self, job: dict[str, Any], rank: int, conn: Connection, cancel_flag: Any):
        self.request = job["request"]
        self.seed = job["seed"]
        self.directory = Path(job["directory"])
        self._rank = rank
        self._conn = conn
        self._cancel_flag = cancel_flag

    def report_progress(self, step: int, total_steps: int) -> None:
        if self._rank == 0:
            self._conn.send(("progress", step, total_steps))

    @property
    def cancel_requested(self) -> bool:
        import torch

        flag = torch.tensor([self._cancel_flag.value if self._rank == 0 else 0], dtype=torch.uint8)
        torch.distributed.broadcast(flag, src=0)
        return bool(flag.item())


def _serve_jobs(backend: Any, rank: int, conn: Connection, cancel_flag: Any) -> None:
    while True:
        job = conn.recv()
        if job is None:
            return
        generation = _WorkerGeneration(job, rank, conn, cancel_flag)
        try:
            reply = ("complete", backend.run(job["kwargs"], generation))
        except GMSError as e:
            reply = ("failed", e.code, e.message, e.pointer)
        # The helpers shared with `run` raise SystemExit, which must not end the worker.
        except (Exception, SystemExit) as e:
            logger.error(f"generation failed on rank {rank}", exc_info=True)
            reply = ("failed", "internal", f"{type(e).__name__}: {e}", None)
        if rank == 0:
            conn.send(reply)


def _worker_main(
    rank: int, world_size: int, port: int, args: Namespace, served: ServedModel, conn: Connection, cancel_flag: Any
) -> None:
    import torch

    # The server process decides when workers stop, so Ctrl-C in the terminal must not interrupt them mid-call.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    if args.log_level is not None:
        logging.set_verbosity(logging.get_log_levels_dict()[args.log_level])
    # Loading applies `--context-parallel`, which joins the process group of all ranks.
    backend = DiffusersBackend(args, served, save_outputs=rank == 0)
    conn.send(("ready",))
    try:
        _serve_jobs(backend, rank, conn, cancel_flag)
    finally:
        torch.distributed.destroy_process_group()


class ParallelBackend:
    def __init__(self, args: Namespace, served: ServedModel):
        import torch

        world_size = torch.cuda.device_count()
        if world_size < 2:
            raise SystemExit(f"--context-parallel needs at least 2 visible GPUs, found {world_size}.")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        self._served = served
        self._failure: str | None = None
        self._closing = False
        # Spawned rather than forked: a forked child inherits whatever CUDA state its parent has.
        context = multiprocessing.get_context("spawn")
        # Lock-free: a worker killed while holding the lock would leave the server blocked on its next write.
        self._cancel_flag = context.Value("b", 0, lock=False)
        self._conns: list[Connection] = []
        self._processes = []
        for rank in range(world_size):
            parent_conn, child_conn = context.Pipe()
            process = context.Process(
                target=_worker_main,
                args=(rank, world_size, port, args, served, child_conn, self._cancel_flag),
                name=f"diffusers-serve-rank-{rank}",
                daemon=True,
            )
            process.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._processes.append(process)

        logger.info(f"loading the pipeline on {world_size} GPUs...")
        self._wait_until_ready()
        threading.Thread(target=self._monitor, name="diffusers-serve-workers", daemon=True).start()

    def _wait_until_ready(self) -> None:
        waiting = dict(zip(self._conns, self._processes))
        while waiting:
            sentinels = {process.sentinel: process for process in waiting.values()}
            for ready in wait([*waiting, *sentinels]):
                process = waiting.pop(ready, None) if isinstance(ready, Connection) else sentinels[ready]
                try:
                    if isinstance(ready, Connection) and ready.recv() == ("ready",):
                        continue
                except EOFError:
                    pass
                self.close()
                raise SystemExit(
                    f"Context-parallel worker {process.name} exited with code {process.exitcode} while loading "
                    "the pipeline. Its error is in the log above."
                )

    def _monitor(self) -> None:
        sentinels = {process.sentinel: process for process in self._processes}
        dead = sentinels[wait(list(sentinels))[0]]
        if self._closing:
            return
        self._failure = f"context-parallel worker {dead.name} exited with code {dead.exitcode}"
        logger.error(f"{self._failure}; the remaining workers are stopped and the server no longer generates.")
        # The other ranks would wait forever in their next collective, so the group is not reused.
        self.close()

    def failure(self) -> str | None:
        return self._failure

    def generate(self, generation: Generation) -> dict[str, list[dict[str, Any]]]:
        if self._failure is not None:
            raise GMSError("internal", self._failure)
        job = {
            "request": generation.request,
            "kwargs": load_request_media(self._served, generation.request),
            "seed": generation.seed,
            "directory": str(generation.directory),
        }
        self._cancel_flag.value = 0
        for conn in self._conns:
            conn.send(job)

        rank_0 = self._conns[0]
        while True:
            if generation.cancel_requested:
                self._cancel_flag.value = 1
            try:
                if not rank_0.poll(_POLL_SECONDS):
                    if self._failure is not None:
                        raise GMSError("internal", self._failure)
                    continue
                message = rank_0.recv()
            except (EOFError, OSError) as e:
                # The monitor may not have noticed yet; rank 0's pipe only breaks once it has exited.
                self._processes[0].join()
                if self._failure is None:
                    self._failure = (
                        f"context-parallel worker {self._processes[0].name} exited with code "
                        f"{self._processes[0].exitcode}"
                    )
                raise GMSError("internal", self._failure) from e
            if message[0] == "progress":
                generation.report_progress(*message[1:])
            elif message[0] == "complete":
                return message[1]
            else:
                _, code, text, pointer = message
                raise GMSError(code, text, pointer)

    def close(self) -> None:
        self._closing = True
        for conn in self._conns:
            try:
                conn.send(None)
            except OSError:
                pass
        for process in self._processes:
            process.join(timeout=_SHUTDOWN_SECONDS)
            if process.is_alive():
                process.kill()
                process.join()
