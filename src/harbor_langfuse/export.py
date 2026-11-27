"""One finished Harbor trial -> Langfuse: a trace that is an experiment item result, plus scores.

Langfuse builds an experiment run from observations that carry the experiment
attributes the Langfuse SDK's run_experiment sets (langfuse.experiment.*), so
nothing else has to be created: the trial's trace *is* the item's result.

    dataset       the trial's Harbor dataset (result.json "source"), or harbor/<job name>
    dataset item  one per task, id derived from dataset and task name (stable across jobs)
    experiment    one run per job, agent, model and dataset
    trace         the agent trajectory (harbor-atif2otel's conversion), root = AGENT span
    scores        each numeric verifier reward on the trace's root observation
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from opentelemetry.proto.common.v1.common_pb2 import AnyValue, ArrayValue, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span, Status

from .client import LangfuseClient

NAMESPACE = uuid.UUID("9f3c7a52-6a7e-4f0e-9a43-5e2f7b6d0c11")
EXPERIMENT_ENVIRONMENT = "sdk-experiment"  # the environment of Langfuse SDK experiment traces
USAGE_PREFIXES = ("llm.token_count.", "llm.cost.", "gen_ai.usage.")
LLM_KINDS = {"LLM"}
MIN_TIME = 946_684_800 * 10**9  # 2000-01-01: an earlier span time means the trajectory had no timestamp

logger = logging.getLogger("harbor_langfuse")


@dataclass
class TrialExport:
    """What was sent for one trial."""

    trial: str
    task: str
    dataset: str
    item_id: str
    experiment_id: str
    experiment_name: str
    trace_id: str = ""  # empty for a trial that was only summarized, not sent
    root_observation_id: str = ""
    rewards: dict = field(default_factory=dict)
    error: str | None = None
    agent: str = ""
    model: str = ""


# ---------------------------------------------------------------- ids and names

def item_id_for(dataset: str, task: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{dataset}\n{task}"))


def experiment_id_for(job_id: str, agent: str, model: str, dataset: str) -> str:
    # 16 hex characters, like the ids the Langfuse SDK gives experiment runs. A run belongs to one
    # dataset, so a job over several Harbor datasets gives one run per dataset.
    return hashlib.sha256(f"{job_id}\n{agent}\n{model}\n{dataset}".encode()).hexdigest()[:16]


def dataset_for(result: dict, job_name: str) -> str:
    source = (result.get("source") or "").split("@")[0]  # without the version: one dataset across versions
    if "/" in source:
        return source
    return f"harbor/{source or job_name}"


def short_model(model: str | None) -> str:
    return (model or "default").split("/")[-1]


# ---------------------------------------------------------------- spans

def _kv(key: str, value) -> KeyValue:
    if isinstance(value, bool):
        return KeyValue(key=key, value=AnyValue(bool_value=value))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return KeyValue(key=key, value=AnyValue(double_value=float(value)))
    if isinstance(value, (list, tuple)):
        return KeyValue(key=key, value=AnyValue(array_value=ArrayValue(values=[AnyValue(string_value=str(v)) for v in value])))
    return KeyValue(key=key, value=AnyValue(string_value=value if isinstance(value, str) else json.dumps(value, default=str)))


def _attr(span: Span, key: str) -> str | None:
    for a in span.attributes:
        if a.key == key:
            return a.value.string_value
    return None


def _set(span: Span, key: str, value) -> None:
    keep = [a for a in span.attributes if a.key != key]
    del span.attributes[:]
    span.attributes.extend(keep + [_kv(key, value)])


def _drop_usage(span: Span) -> None:
    """Agent and chain spans repeat the usage and cost of their model calls; Langfuse would count it twice."""
    keep = [a for a in span.attributes if not a.key.startswith(USAGE_PREFIXES)]
    del span.attributes[:]
    span.attributes.extend(keep)


def _nanos(ts: str | None) -> int:
    if not ts:
        return 0
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


def _trial_times(result: dict) -> tuple[int, int]:
    """When the agent ran, for spans the trajectory gives no time; now if the result has none either."""
    execution = result.get("agent_execution") or {}
    start = _nanos(execution.get("started_at") or result.get("started_at")) or time.time_ns()
    end = _nanos(execution.get("finished_at") or result.get("finished_at"))
    return start, max(start, end)


def _error_message(exc: dict, limit: int) -> str:
    return f"{exc.get('exception_type')}: {str(exc.get('exception_message') or '')}"[:limit]


def _fallback_spans(result: dict, trial: str) -> ResourceSpans:
    """A trial without a trajectory (it failed before or during the agent run): one root span."""
    trace_id = hashlib.sha256(trial.encode()).digest()[:16]
    span_id = hashlib.sha256(("root\n" + trial).encode()).digest()[:8]
    agent = (result.get("config") or {}).get("agent") or {}
    start, end = _trial_times(result)
    span = Span(trace_id=trace_id, span_id=span_id, name=agent.get("name") or "agent",
                start_time_unix_nano=start, end_time_unix_nano=end,
                attributes=[_kv("openinference.span.kind", "AGENT")])
    return ResourceSpans(resource=Resource(attributes=[_kv("service.name", "harbor")]),
                         scope_spans=[ScopeSpans(spans=[span])])


def trial_spans(trial_dir: Path, result: dict) -> ResourceSpans:
    traj_path = trial_dir / "agent" / "trajectory.json"
    if traj_path.exists():
        try:
            from harbor_atif2otel.convert import convert_trajectory

            rs = convert_trajectory(json.loads(traj_path.read_text()), trace_seed=trial_dir.name)
            if any(ss.spans for ss in rs.scope_spans):
                return rs
        except Exception as exc:  # noqa: BLE001 - an unreadable trajectory still gets a root span
            logger.warning("harbor-langfuse: could not convert %s, sending the trial without its trajectory: %r",
                           traj_path, exc)
    return _fallback_spans(result, trial_dir.name)


# ---------------------------------------------------------------- one trial

def task_instruction(trial_dir: Path) -> str | None:
    """The task's instruction.md, from the task directory Harbor resolved for the trial."""
    try:
        from harbor.models.trial.config import TrialConfig

        config = TrialConfig.model_validate_json((trial_dir / "config.json").read_text())
        path = Path(config.task.get_local_path()) / "instruction.md"
        return path.read_text().strip() or None
    except Exception:  # noqa: BLE001 - fall back to the trajectory's first user message
        return None


def numeric_rewards(result: dict) -> dict[str, float]:
    rewards = ((result.get("verifier_result") or {}).get("rewards")) or {}
    return {k: float(v) for k, v in rewards.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def summarize_trial(trial_dir: Path, *, job_name: str, job_id: str | None = None, dataset: str | None = None,
                    run_name: str | None = None) -> TrialExport:
    """Where a finished trial belongs in Langfuse and how it scored, read from its result.json; sends nothing."""
    return _summarize(json.loads((trial_dir / "result.json").read_text()), trial_dir, job_name, job_id, dataset,
                      run_name)


def _summarize(result: dict, trial_dir: Path, job_name: str, job_id: str | None, dataset: str | None,
               run_name: str | None) -> TrialExport:
    config = result.get("config") or {}
    agent_cfg = config.get("agent") or {}
    agent, model = agent_cfg.get("name") or "agent", agent_cfg.get("model_name") or ""
    task = result.get("task_name") or trial_dir.name.split("__")[0]
    dataset = dataset or dataset_for(result, job_name)
    job_id = job_id or config.get("job_id") or job_name
    return TrialExport(trial=trial_dir.name, task=task, dataset=dataset, item_id=item_id_for(dataset, task),
                       experiment_id=experiment_id_for(job_id, agent, model, dataset),
                       experiment_name=run_name or f"{job_name} ({agent}, {short_model(model)})",
                       rewards=numeric_rewards(result), error=(result.get("exception_info") or {}).get("exception_type"),
                       agent=agent, model=model)


def _ensure_item(client: LangfuseClient, ex: TrialExport, task_text: str | None, first_message: str | None,
                 metadata: dict) -> None:
    """Creates the task's dataset item, or updates it when the task changed. The instruction is the task's
    instruction.md; the trajectory's first user message stands in for it only on an item that has none,
    because it differs per agent. A trial that knows neither must not overwrite an item that has one."""
    existing = client.get_item(ex.item_id)
    if existing is not None:
        current = existing.get("input")
        has_instruction = isinstance(current, dict) and bool(current.get("instruction"))
        if task_text:
            if current == {"task": ex.task, "instruction": task_text} and existing.get("metadata") == metadata:
                return
        elif has_instruction or not first_message:
            return
    instruction = task_text or first_message
    client.upsert_item(ex.item_id, ex.dataset,
                       {"task": ex.task, "instruction": instruction} if instruction else {"task": ex.task}, metadata)


def export_trial(client: LangfuseClient, trial_dir: Path, *, job_name: str, job_id: str | None = None,
                 dataset: str | None = None, run_name: str | None = None, tags: list[str] | None = None,
                 ensure_dataset=None) -> TrialExport:
    """Sends one finished trial to Langfuse and returns what was sent."""
    result = json.loads((trial_dir / "result.json").read_text())
    ex = _summarize(result, trial_dir, job_name, job_id, dataset, run_name)
    exc = result.get("exception_info") or {}

    rs = trial_spans(trial_dir, result)
    spans = [s for ss in rs.scope_spans for s in ss.spans]
    root = next((s for s in spans if not s.parent_span_id), spans[0])
    instruction = _attr(root, "input.value")
    start, end = _trial_times(result)
    for s in spans:
        if _attr(s, "openinference.span.kind") not in LLM_KINDS:
            _drop_usage(s)
        if s.start_time_unix_nano < MIN_TIME:  # a step without a timestamp would put the trace in 1970
            s.start_time_unix_nano += start  # keeps the converter's small offsets between such spans
        if s.end_time_unix_nano < MIN_TIME:
            s.end_time_unix_nano += start
        s.end_time_unix_nano = max(s.end_time_unix_nano, s.start_time_unix_nano)
    if exc:
        root.status.CopyFrom(Status(code=Status.STATUS_CODE_ERROR, message=_error_message(exc, 1000)))

    dataset_id = (ensure_dataset or (lambda name: client.ensure_dataset(name)))(ex.dataset)
    _ensure_item(client, ex, task_instruction(trial_dir), instruction,
                 {"harbor_task": ex.task, "harbor_dataset": result.get("source"),
                  "task_checksum": result.get("task_checksum")})

    meta = {"harbor_job": job_name, "harbor_trial": ex.trial, "harbor_task": ex.task, "agent": ex.agent,
            "model": ex.model, "agent_version": (result.get("agent_info") or {}).get("version"),
            **{f"reward_{k}": v for k, v in ex.rewards.items()}}
    if exc:
        meta["exception"] = exc.get("exception_type")
    _set(root, "langfuse.trace.name", ex.task)
    for k, v in meta.items():
        if v is not None:
            _set(root, f"langfuse.trace.metadata.{k}", v)
    _set(root, "langfuse.trace.tags", ["harbor", ex.agent, *(tags or [])])
    ex.trace_id, ex.root_observation_id = root.trace_id.hex(), root.span_id.hex()
    experiment_attrs = {
        "langfuse.environment": EXPERIMENT_ENVIRONMENT,
        "langfuse.experiment.id": ex.experiment_id,
        "langfuse.experiment.name": ex.experiment_name,
        "langfuse.experiment.dataset.id": dataset_id,
        "langfuse.experiment.item.id": ex.item_id,
        "langfuse.experiment.item.root_observation_id": ex.root_observation_id,
        "langfuse.experiment.metadata.harbor_job": job_name,
        "langfuse.experiment.metadata.agent": ex.agent,
        "langfuse.experiment.metadata.model": ex.model,
    }
    for s in spans:  # like the SDK: every observation of the item carries the experiment attributes
        for k, v in experiment_attrs.items():
            _set(s, k, v)
    client.send_spans(rs)

    score = {"traceId": ex.trace_id, "observationId": ex.root_observation_id, "environment": EXPERIMENT_ENVIRONMENT}
    for name, value in ex.rewards.items():
        client.score({**score, "id": str(uuid.uuid5(NAMESPACE, f"{ex.trial}\n{name}")), "name": name, "value": value,
                      "metadata": {"harbor_trial": ex.trial, "harbor_task": ex.task}})
    if exc:
        client.score({**score, "id": str(uuid.uuid5(NAMESPACE, f"{ex.trial}\ntrial_error")), "name": "trial_error",
                      "value": 1, "dataType": "BOOLEAN", "comment": _error_message(exc, 500)})
    return ex


def run_scores(client: LangfuseClient, exports: list[TrialExport]) -> list[dict]:
    """Run-level scores per experiment: the mean of each reward over the run's trials, and the error rate."""
    out = []
    for run_id in sorted({e.experiment_id for e in exports}):
        mine = [e for e in exports if e.experiment_id == run_id]
        names = sorted({k for e in mine for k in e.rewards})
        values = {n: [e.rewards[n] for e in mine if n in e.rewards] for n in names}
        scores = {f"mean_{n}": sum(v) / len(v) for n, v in values.items() if v}
        scores["error_rate"] = sum(1 for e in mine if e.error) / len(mine)
        for name, value in scores.items():
            body = {"id": str(uuid.uuid5(NAMESPACE, f"{run_id}\n{name}")), "datasetRunId": run_id, "name": name,
                    "value": value, "metadata": {"trials": len(mine)}}
            client.score(body)
            out.append(body)
    return out


def job_name_of(job_dir: Path) -> str:
    cfg = job_dir / "config.json"
    if cfg.exists():
        return json.loads(cfg.read_text()).get("job_name") or job_dir.name
    return job_dir.name


def trial_dirs(job_dir: Path) -> list[Path]:
    return sorted(d for d in job_dir.iterdir() if d.is_dir() and (d / "result.json").exists())


__all__ = ["TrialExport", "export_trial", "summarize_trial", "run_scores", "item_id_for", "experiment_id_for", "trial_dirs", "job_name_of"]
