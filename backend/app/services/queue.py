"""
FIFO job queue with bounded concurrency for local hardware.

Runs heavy pipelines one-at-a-time by default (configurable). Prevents
simultaneous LLM/GPU work that would exhaust 8GB RAM on a GTX 1650 box.
"""

from __future__ import annotations

import asyncio
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Awaitable, Callable

from ..settings import settings
from . import jobs, notify

PipelineFn = Callable[..., Awaitable[None]]


@dataclass
class QueueItem:
    job_id: str
    product: str
    fn: PipelineFn
    args: tuple
    kwargs: dict
    enqueued_at: datetime = field(default_factory=datetime.now)


class JobQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[QueueItem | None] = asyncio.Queue()
        self._pending: deque[str] = deque()
        self._running: dict[str, asyncio.Task] = {}
        self._sem = asyncio.Semaphore(max(1, settings.QUEUE_MAX_CONCURRENT))
        self._dispatcher: asyncio.Task | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._dispatcher = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        if not self._started:
            return
        await self._queue.put(None)
        if self._dispatcher:
            await self._dispatcher
        running = list(self._running.values())
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
        self._started = False

    async def enqueue(
        self,
        job_id: str,
        product: str,
        fn: PipelineFn,
        *args,
        **kwargs,
    ) -> None:
        await self.start()
        self._pending.append(job_id)
        jobs.update(
            job_id,
            state="queued",
            progress=0,
            message="Waiting in queue…",
            product=product,
        )
        await self._queue.put(QueueItem(job_id, product, fn, args, kwargs))

    def position(self, job_id: str) -> int:
        """1-based position in the pending queue, or 0 if not waiting."""
        try:
            return list(self._pending).index(job_id) + 1
        except ValueError:
            return 0

    def snapshot(self) -> dict:
        return {
            "max_concurrent": settings.QUEUE_MAX_CONCURRENT,
            "pending": list(self._pending),
            "running": list(self._running.keys()),
            "pending_count": len(self._pending),
            "running_count": len(self._running),
        }

    async def cancel(self, job_id: str) -> bool:
        status = jobs.load(job_id)
        if status is None:
            return False
        if job_id in self._running:
            self._running[job_id].cancel()
            return True
        if job_id in self._pending:
            jobs.update(job_id, state="failed", message="Cancelled", error="Cancelled by user")
            try:
                self._pending.remove(job_id)
            except ValueError:
                pass
            return True
        return False

    async def _dispatch_loop(self) -> None:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            if item.job_id in self._pending:
                try:
                    self._pending.remove(item.job_id)
                except ValueError:
                    pass
            task = asyncio.create_task(self._run_item(item))
            self._running[item.job_id] = task

    async def _run_item(self, item: QueueItem) -> None:
        job_id = item.job_id
        try:
            async with self._sem:
                current = jobs.load(job_id)
                if current and current.state == "failed" and current.error == "Cancelled by user":
                    return
                jobs.update(job_id, state="processing", message="Processing started…")
                timeout = max(1, settings.QUEUE_JOB_TIMEOUT_SECONDS)
                await asyncio.wait_for(
                    item.fn(*item.args, **item.kwargs), timeout=timeout
                )
                # Email the user when it's ready (no-op without SMTP).
                done = jobs.load(job_id)
                if done and done.user_email and done.state == "done":
                    label = notify.product_label(done.product)
                    await notify.send(
                        done.user_email, f"Your {label} is ready",
                        f"Good news — your {label} is ready. "
                        f"Open ChatStory and go to 'My stuff' to preview and download it.")
        except asyncio.TimeoutError:
            jobs.update(
                job_id,
                state="failed",
                message="Processing timed out",
                error=f"Job exceeded the {settings.QUEUE_JOB_TIMEOUT_SECONDS}s time limit",
            )
        except asyncio.CancelledError:
            jobs.update(
                job_id,
                state="failed",
                message="Cancelled",
                error="Cancelled",
            )
        except Exception as exc:
            tb = traceback.format_exc()
            jobs.update(
                job_id,
                state="failed",
                message="Processing failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            print(f"Queue job {job_id} failed:\n{tb}")
        finally:
            self._running.pop(job_id, None)


job_queue = JobQueue()
