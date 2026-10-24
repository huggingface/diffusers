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

"""The generation resource and the single-worker queue that runs generations one at a time."""

from __future__ import annotations

import queue
import secrets
import shutil
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ...utils import logging
from .manifest import GMSError, ResolvedRequest


logger = logging.get_logger("diffusers-cli/serve")

EXPIRY_SECONDS = 3600


class Generation:
    def __init__(
        self,
        model: str,
        revision: str | None,
        request: ResolvedRequest,
        seeded: bool,
        default_seed: int | None = None,
    ):
        self.id = f"gen_{uuid.uuid4().hex[:12]}"
        self.model = model
        self.revision = revision
        self.request = request
        # The spec has the server pick the seed when the request omits it, so the result is reproducible.
        self.seed = None
        if seeded and request.seed is not None:
            self.seed = request.seed
        elif seeded and default_seed is not None:
            self.seed = default_seed
        elif seeded:
            self.seed = secrets.randbelow(2**32)
        self.status = "queued"
        self.created_at = datetime.now(timezone.utc)
        self.expires_at: datetime | None = None
        self.progress: dict[str, Any] | None = None
        self.outputs: dict[str, list[dict[str, Any]]] | None = None
        self.usage: dict[str, Any] | None = None
        self.error: GMSError | None = None
        self.cancel_requested = False
        self.directory: Path | None = None
        self.events: list[tuple[str, dict[str, Any] | None]] = []
        self.done = False
        self._condition = threading.Condition()

    def report_progress(self, step: int, total_steps: int) -> None:
        self.progress = {"step": step, "total_steps": total_steps, "fraction": round(step / total_steps, 4)}
        self._emit("progress", self.progress)

    def wait_for_events(self, index: int, timeout: float) -> tuple[list[tuple[str, dict[str, Any] | None]], bool]:
        with self._condition:
            self._condition.wait_for(lambda: len(self.events) > index or self.done, timeout)
            return self.events[index:], self.done

    def wait(self, timeout: float) -> None:
        with self._condition:
            self._condition.wait_for(lambda: self.done, timeout)

    def finish(self, status: str, error: GMSError | None = None) -> None:
        with self._condition:
            self.status = status
            self.error = error
            self.progress = None
            self.expires_at = datetime.now(timezone.utc) + timedelta(seconds=EXPIRY_SECONDS)
            if error is not None:
                self.events.append(("error", {"code": error.code, "message": error.message}))
            elif status == "complete":
                self.events.append(("complete", None))
            self.done = True
            self._condition.notify_all()

    def _emit(self, event: str, data: dict[str, Any] | None) -> None:
        with self._condition:
            self.events.append((event, data))
            self._condition.notify_all()


class GenerationQueue:
    def __init__(self, backend: Any, output_dir: str):
        self._backend = backend
        self._output_dir = Path(output_dir)
        self._generations: dict[str, Generation] = {}
        self._lock = threading.Lock()
        self._pending: queue.Queue[Generation] = queue.Queue()
        threading.Thread(target=self._work, name="diffusers-serve-worker", daemon=True).start()

    def failure(self) -> str | None:
        return self._backend.failure()

    def submit(self, generation: Generation) -> None:
        self._prune()
        with self._lock:
            self._generations[generation.id] = generation
        self._pending.put(generation)

    def get(self, generation_id: str) -> Generation:
        self._prune()
        with self._lock:
            generation = self._generations.get(generation_id)
        if generation is None:
            raise GMSError("not_found", f"no generation with id {generation_id!r}")
        return generation

    def cancel(self, generation_id: str) -> Generation:
        generation = self.get(generation_id)
        with self._lock:
            if generation.done:
                return generation
            generation.cancel_requested = True
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

    def _remove_files(self, generation: Generation) -> None:
        if generation.directory is not None:
            shutil.rmtree(generation.directory, ignore_errors=True)

    def _work(self) -> None:
        while True:
            generation = self._pending.get()
            with self._lock:
                if generation.done:
                    continue
                generation.status = "generating"
            generation.directory = self._output_dir / generation.id

            started = time.perf_counter()
            error = None
            try:
                generation.directory.mkdir(parents=True, exist_ok=True)
                outputs = self._backend.generate(generation)
            except GMSError as e:
                error = e
            # The helpers shared with `run` raise SystemExit, and anything escaping here would end the
            # only worker and leave every later generation queued.
            except BaseException as e:
                logger.error(f"generation {generation.id} failed", exc_info=True)
                error = GMSError("internal", f"{type(e).__name__}: {e}")

            with self._lock:
                deleted = generation.id not in self._generations
                if error is not None:
                    generation.finish("failed", error)
                elif generation.cancel_requested:
                    generation.finish("canceled")
                else:
                    generation.outputs = outputs
                    generation.usage = {"compute_seconds": round(time.perf_counter() - started, 2)}
                    generation.finish("complete")
            if deleted or generation.status != "complete":
                self._remove_files(generation)
