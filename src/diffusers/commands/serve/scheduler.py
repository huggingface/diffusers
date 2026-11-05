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

"""Generation jobs, and the scheduler that runs them in arrival order, one at a time on each backend."""

from __future__ import annotations

import asyncio
import queue
import shutil
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from ...utils import logging
from .manifest import GMSError, ResolvedRequest


logger = logging.get_logger("diffusers-cli/serve")

EXPIRY_SECONDS = 3600


class GenerationJob:
    """One generation, from the request that created it to its outputs: the resource `/v1/generations/{id}` returns."""

    def __init__(self, request: ResolvedRequest):
        self.id = f"gen_{uuid.uuid4().hex[:12]}"
        self.request = request
        self.status = "queued"
        self.created_at = datetime.now(timezone.utc)
        self.expires_at: datetime | None = None
        self.progress: dict[str, Any] | None = None
        self.outputs: dict[str, list[dict[str, Any]]] | None = None
        self.usage: dict[str, Any] | None = None
        self.error: GMSError | None = None
        self.canceled = False
        # When a running generation must stop, from `--generation-timeout`; None when it has no limit.
        self.deadline: float | None = None
        self.directory: Path | None = None
        self.events: list[tuple[str, dict[str, Any] | None]] = []
        self.done = False
        self._lock = threading.Lock()
        # Requests waiting on the event loop for the job to change. The worker thread wakes them through their loop.
        self._waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []

    @property
    def cancel_requested(self) -> bool:
        """Whether the pipeline should stop at its next step: a client canceled it, or it ran past its deadline."""
        return self.canceled or (self.deadline is not None and time.monotonic() > self.deadline)

    def report_progress(self, step: int, total_steps: int) -> None:
        with self._lock:
            self.progress = {"step": step, "total_steps": total_steps, "fraction": round(step / total_steps, 4)}
            self.events.append(("progress", self.progress))
            self._notify_waiting_requests()

    async def wait_for_events(
        self, index: int, timeout: float
    ) -> tuple[list[tuple[str, dict[str, Any] | None]], bool]:
        await self._wait_until(lambda: len(self.events) > index or self.done, timeout)
        # Read together, so a stream never sees `done` without the final event that came with it.
        with self._lock:
            return self.events[index:], self.done

    async def wait(self, timeout: float) -> None:
        await self._wait_until(lambda: self.done, timeout)

    def finish(self, status: str, error: GMSError | None = None) -> None:
        with self._lock:
            self.status = status
            self.error = error
            self.progress = None
            self.expires_at = datetime.now(timezone.utc) + timedelta(seconds=EXPIRY_SECONDS)
            if error is not None:
                self.events.append(("error", {"code": error.code, "message": error.message}))
            elif status == "complete":
                self.events.append(("complete", None))
            self.done = True
            self._notify_waiting_requests()

    async def _wait_until(self, condition: Callable[[], bool], timeout: float) -> None:
        # Waiting on the event loop, not a pool thread, so any number of requests can wait at once.
        loop = asyncio.get_running_loop()
        give_up_at = loop.time() + timeout
        while True:
            event = asyncio.Event()
            with self._lock:
                if condition():
                    return
                self._waiters.append((loop, event))
            try:
                await asyncio.wait_for(event.wait(), give_up_at - loop.time())
            except asyncio.TimeoutError:
                return
            finally:
                with self._lock:
                    self._waiters.remove((loop, event))

    def _notify_waiting_requests(self) -> None:
        for loop, event in self._waiters:
            # A loop closed at shutdown has no request left to wake, and raising here would end the worker.
            if not loop.is_closed():
                loop.call_soon_threadsafe(event.set)


class Scheduler:
    """Runs generation jobs first in, first out, on one worker thread per backend.

    It also keeps every job until it expires, so the API can look jobs up, cancel them and delete them.
    """

    def __init__(self, backends: list[Any], output_dir: str, timeout: float | None = None):
        self._backends = backends
        self._output_dir = Path(output_dir)
        self._timeout = timeout
        self._generations: dict[str, GenerationJob] = {}
        self._lock = threading.Lock()
        self._pending: queue.Queue[GenerationJob] = queue.Queue()
        # The generations a worker is running now.
        self._running: set[GenerationJob] = set()
        # Every worker takes the next job from the same queue, so no job waits behind a long one on a busy backend.
        for index, backend in enumerate(backends):
            threading.Thread(
                target=self._work, args=(backend,), name=f"diffusers-serve-worker-{index}", daemon=True
            ).start()

    def failure(self) -> str | None:
        for backend in self._backends:
            failure = backend.failure()
            if failure is not None:
                return failure
        if self._timeout is None:
            return None
        # A timed out generation stops at its next step. One still running long after that is stuck inside a step,
        # or in a pipeline without step callbacks, and its worker takes no more jobs.
        now = time.monotonic()
        with self._lock:
            for generation in self._running:
                if now > generation.deadline + self._timeout:
                    return (
                        f"generation {generation.id} has run for over twice the --generation-timeout of "
                        f"{self._timeout:g}s and did not stop when interrupted"
                    )
        return None

    def busy(self) -> bool:
        return bool(self._running) or not self._pending.empty()

    def submit(self, generation: GenerationJob) -> None:
        self._prune()
        with self._lock:
            self._generations[generation.id] = generation
        self._pending.put(generation)

    def get(self, generation_id: str) -> GenerationJob:
        self._prune()
        with self._lock:
            generation = self._generations.get(generation_id)
        if generation is None:
            raise GMSError("not_found", f"no generation with id {generation_id!r}")
        return generation

    def cancel(self, generation_id: str) -> GenerationJob:
        generation = self.get(generation_id)
        with self._lock:
            if generation.done:
                return generation
            generation.canceled = True
            if generation.status == "queued":
                generation.finish("canceled")
        return generation

    def delete(self, generation_id: str) -> None:
        generation = self.cancel(generation_id)
        with self._lock:
            del self._generations[generation.id]
        # A generation still running keeps writing into its directory; the worker removes it once the backend returns.
        if generation.done:
            self._remove_files(generation)

    def _prune(self) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            expired = [g for g in self._generations.values() if g.expires_at is not None and g.expires_at <= now]
            for generation in expired:
                del self._generations[generation.id]
        for generation in expired:
            self._remove_files(generation)

    def _remove_files(self, generation: GenerationJob) -> None:
        if generation.directory is not None:
            shutil.rmtree(generation.directory, ignore_errors=True)

    def _work(self, backend: Any) -> None:
        while True:
            generation = self._pending.get()
            with self._lock:
                if generation.done:
                    continue
                generation.status = "generating"
                if self._timeout is not None:
                    generation.deadline = time.monotonic() + self._timeout
                self._running.add(generation)
            generation.directory = self._output_dir / generation.id

            started = time.perf_counter()
            error = None
            try:
                generation.directory.mkdir(parents=True, exist_ok=True)
                outputs = backend.generate(generation)
            except GMSError as e:
                error = e
            # The helpers shared with `run` raise SystemExit, and anything escaping here would end this
            # worker and leave its backend idle for good.
            except BaseException as e:
                logger.error(f"generation {generation.id} failed", exc_info=True)
                error = GMSError("internal", f"{type(e).__name__}: {e}")
            returned_at = time.monotonic()

            with self._lock:
                deleted = generation.id not in self._generations
                # Before `finish` wakes any waiter, so a finished job is never reported while the server looks busy.
                self._running.discard(generation)
                if error is not None:
                    generation.finish("failed", error)
                elif generation.deadline is not None and returned_at > generation.deadline:
                    message = f"generation exceeded the --generation-timeout of {self._timeout:g}s"
                    generation.finish("failed", GMSError("internal", message))
                elif generation.canceled:
                    generation.finish("canceled")
                else:
                    generation.outputs = outputs
                    generation.usage = {"compute_seconds": round(time.perf_counter() - started, 2)}
                    generation.finish("complete")
            if deleted or generation.status != "complete":
                self._remove_files(generation)
