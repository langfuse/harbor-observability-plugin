"""Harbor job plugin: harbor run ... --plugin langfuse

Options (--pk key=value): dataset, run_name, tags (comma-separated), base_url.
Credentials: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from harbor.models.job.plugin import BaseJobPlugin

from .client import LangfuseClient, LangfuseError
from .export import TrialExport, export_trial, run_scores

logger = logging.getLogger("harbor_langfuse")


class LangfusePlugin(BaseJobPlugin):
    def __init__(self, *, dataset: str | None = None, run_name: str | None = None, tags: str | list | None = None,
                 base_url: str | None = None, public_key: str | None = None, secret_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._dataset, self._run_name = dataset, run_name
        self._tags = [t for t in (tags.split(",") if isinstance(tags, str) else tags or []) if t]
        self._client = LangfuseClient(base_url, public_key, secret_key)
        self._datasets: dict[str, str] = {}
        self._lock = threading.Lock()
        self._exports: list[TrialExport] = []
        self._failures = 0
        self._job_dir: Path | None = None
        self._job_name = ""
        self._job_id: str | None = None

    async def on_job_start(self, job) -> None:
        self._job_dir, self._job_name = job.job_dir, job.config.job_name
        self._job_id = str(getattr(job, "id", "") or "") or None
        await asyncio.to_thread(self._client.check)  # fail early on wrong keys
        job.on_trial_ended(self._on_trial_ended)

    def _ensure_dataset(self, name: str) -> str:
        with self._lock:
            if name not in self._datasets:
                self._datasets[name] = self._client.ensure_dataset(
                    name, description="Harbor tasks, one item per task (created by harbor-langfuse)")
            return self._datasets[name]

    async def _on_trial_ended(self, event) -> None:
        trial_dir = self._job_dir / event.config.trial_name
        try:
            export = await asyncio.to_thread(
                export_trial, self._client, trial_dir, job_name=self._job_name, job_id=self._job_id,
                dataset=self._dataset, run_name=self._run_name, tags=self._tags, ensure_dataset=self._ensure_dataset)
            with self._lock:
                self._exports.append(export)
        except (LangfuseError, OSError, ValueError) as exc:  # never fail the job over Langfuse
            self._failures += 1
            logger.warning("harbor-langfuse: could not export %s: %s", trial_dir.name, exc)

    async def on_job_end(self, job_result) -> None:
        if self._exports:
            try:
                await asyncio.to_thread(run_scores, self._client, self._exports)
            except LangfuseError as exc:
                logger.warning("harbor-langfuse: run scores failed: %s", exc)
        runs = sorted({(e.experiment_name, e.dataset) for e in self._exports})
        for name, dataset in runs:
            print(f"harbor-langfuse: experiment run '{name}' on dataset '{dataset}' "
                  f"in {self._client.base_url}", flush=True)
        if self._failures:
            print(f"harbor-langfuse: {self._failures} trials could not be exported (see warnings)", flush=True)
