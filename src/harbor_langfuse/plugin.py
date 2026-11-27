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

from .client import LangfuseClient
from .export import TrialExport, export_trial, run_scores, summarize_trial, trial_dirs

logger = logging.getLogger("harbor_langfuse")


class LangfusePlugin(BaseJobPlugin):
    def __init__(self, *, dataset: str | None = None, run_name: str | None = None, tags: str | list | None = None,
                 base_url: str | None = None, public_key: str | None = None, secret_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        # Harbor parses --pk values as JSON, so "run_name=2026" or "tags=1" arrive as numbers
        self._dataset = str(dataset) if dataset is not None else None
        self._run_name = str(run_name) if run_name is not None else None
        if tags is None:
            tags = []
        elif not isinstance(tags, (list, tuple)):
            tags = str(tags).split(",")
        self._tags = [t for t in (str(t).strip() for t in tags) if t]
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
        except Exception as exc:  # noqa: BLE001 - Harbor awaits this hook in the trial: never fail it over Langfuse
            self._failures += 1
            logger.warning("harbor-langfuse: could not export %s: %r", trial_dir.name, exc)

    def _run_trials(self) -> list[TrialExport]:
        """All trials of the runs this session added to. A resumed job only sends its new trials, but
        the run-level scores are about the whole run, so earlier trials are read from the job directory."""
        sent = {e.trial: e for e in self._exports}
        runs = {e.experiment_id for e in self._exports}
        trials = list(self._exports)
        for trial_dir in trial_dirs(self._job_dir) if self._job_dir and self._job_dir.is_dir() else []:
            if trial_dir.name in sent:
                continue
            try:
                earlier = summarize_trial(trial_dir, job_name=self._job_name, job_id=self._job_id,
                                          dataset=self._dataset, run_name=self._run_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("harbor-langfuse: could not read %s: %r", trial_dir.name, exc)
                continue
            if earlier.experiment_id in runs:
                trials.append(earlier)
        return trials

    async def on_job_end(self, job_result) -> None:
        if self._exports:
            try:
                await asyncio.to_thread(lambda: run_scores(self._client, self._run_trials()))
            except Exception as exc:  # noqa: BLE001
                logger.warning("harbor-langfuse: run scores failed: %r", exc)
        runs = sorted({(e.experiment_name, e.dataset) for e in self._exports})
        for name, dataset in runs:
            print(f"harbor-langfuse: experiment run '{name}' on dataset '{dataset}' "
                  f"in {self._client.base_url}", flush=True)
        if self._failures:
            print(f"harbor-langfuse: {self._failures} trials could not be exported (see warnings)", flush=True)
