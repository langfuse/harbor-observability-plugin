"""One finished Harbor trial -> Langfuse: a trace that is an experiment item result, plus scores.

Langfuse builds an experiment run from observations that carry the experiment
attributes the Langfuse SDK's run_experiment sets (langfuse.experiment.*), so
nothing else has to be created: the trial's trace *is* the item's result.

    dataset       the trial's Harbor dataset (result.json "source"), or harbor/<job name>
    dataset item  one per task, id derived from dataset and task name (stable across jobs)
    experiment    one run per job, agent and model
    trace         the agent trajectory (harbor-atif2otel's conversion), root = AGENT span
    scores        each numeric verifier reward on the trace's root observation
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from opentelemetry.proto.common.v1.common_pb2 import AnyValue, ArrayValue, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span, Status

from .client import LangfuseClient

NAMESPACE = uuid.UUID("9f3c7a52-6a7e-4f0e-9a43-5e2f7b6d0c11")
EXPERIMENT_ENVIRONMENT = "sdk-experiment"  # the environment of Langfuse SDK experiment traces
USAGE_PREFIXES = ("llm.token_count.", "llm.cost.", "gen_ai.usage.")
LLM_KINDS = {"LLM"}


@dataclass
class TrialExport:
    """What was sent for one trial."""

    trial: str
    task: str
    dataset: str
    item_id: str
    experiment_id: str
    experiment_name: str
    trace_id: str
    root_observation_id: str
    rewards: dict = field(default_factory=dict)
    error: str | None = None
    agent: str = ""
    model: str = ""


# ---------------------------------------------------------------- ids and names

def item_id_for(dataset: str, task: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{dataset}\n{task}"))


def experiment_id_for(job_id: str, agent: str, model: str) -> str:
    # 16 hex characters, like the ids the Langfuse SDK gives experiment runs.
    return hashlib.sha256(f"{job_id}\n{agent}\n{model}".encode()).hexdigest()[:16]


def dataset_for(result: dict, job_name: str) -> str:
    source = result.get("source")
    if source and "/" in source:
        return source.split("@")[0]
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
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1e9)


def _fallback_spans(result: dict, trial: str) -> ResourceSpans:
    """A trial without a trajectory (it failed before or during the agent run): one root span."""
    trace_id = hashlib.sha256(trial.encode()).digest()[:16]
    span_id = hashlib.sha256(("root\n" + trial).encode()).digest()[:8]
    agent = (result.get("config") or {}).get("agent") or {}
    exc = result.get("exception_info") or {}
    start = _nanos((result.get("agent_execution") or {}).get("started_at") or result.get("started_at"))
    end = _nanos((result.get("agent_execution") or {}).get("finished_at") or result.get("finished_at")) or start
    span = Span(trace_id=trace_id, span_id=span_id, name=agent.get("name") or "agent",
                start_time_unix_nano=start, end_time_unix_nano=end,
                attributes=[_kv("openinference.span.kind", "AGENT")])
    if exc:
        span.status.CopyFrom(Status(code=Status.STATUS_CODE_ERROR,
                                    message=f"{exc.get('exception_type')}: {exc.get('exception_message', '')}"[:1000]))
    return ResourceSpans(resource=Resource(attributes=[_kv("service.name", "harbor")]),
                         scope_spans=[ScopeSpans(spans=[span])])


def trial_spans(trial_dir: Path, result: dict) -> ResourceSpans:
    traj_path = trial_dir / "agent" / "trajectory.json"
    if traj_path.exists():
        try:
            from harbor_atif2otel.convert import convert_trajectory

            return convert_trajectory(json.loads(traj_path.read_text()), trace_seed=trial_dir.name)
        except Exception:  # noqa: BLE001 - an unreadable trajectory still gets a root span
            pass
    return _fallback_spans(result, trial_dir.name)


# ---------------------------------------------------------------- one trial

def numeric_rewards(result: dict) -> dict[str, float]:
    rewards = ((result.get("verifier_result") or {}).get("rewards")) or {}
    return {k: float(v) for k, v in rewards.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def export_trial(client: LangfuseClient, trial_dir: Path, *, job_name: str, job_id: str | None = None,
                 dataset: str | None = None, run_name: str | None = None, tags: list[str] | None = None,
                 ensure_dataset=None) -> TrialExport:
    """Sends one finished trial to Langfuse and returns what was sent."""
    result = json.loads((trial_dir / "result.json").read_text())
    config = result.get("config") or {}
    agent_cfg = config.get("agent") or {}
    agent, model = agent_cfg.get("name") or "agent", agent_cfg.get("model_name") or ""
    task = result.get("task_name") or trial_dir.name.split("__")[0]
    dataset = dataset or dataset_for(result, job_name)
    job_id = job_id or config.get("job_id") or job_name
    experiment_id = experiment_id_for(job_id, agent, model)
    experiment_name = run_name or f"{job_name} ({agent}, {short_model(model)})"
    rewards = numeric_rewards(result)
    exc = result.get("exception_info") or {}

    rs = trial_spans(trial_dir, result)
    spans = [s for ss in rs.scope_spans for s in ss.spans]
    root = next((s for s in spans if not s.parent_span_id), spans[0])
    user_input = None
    for s in spans:
        if s is root:
            user_input = _attr(s, "input.value")
        if _attr(s, "openinference.span.kind") not in LLM_KINDS:
            _drop_usage(s)

    dataset_id = (ensure_dataset or (lambda name: client.ensure_dataset(name)))(dataset)
    item_id = item_id_for(dataset, task)
    client.upsert_item(item_id, dataset, {"task": task, "instruction": user_input} if user_input else {"task": task},
                       {"harbor_task": task, "harbor_dataset": result.get("source"),
                        "task_checksum": result.get("task_checksum")})

    meta = {"harbor_job": job_name, "harbor_trial": trial_dir.name, "harbor_task": task, "agent": agent,
            "model": model, "agent_version": (result.get("agent_info") or {}).get("version"),
            **{f"reward_{k}": v for k, v in rewards.items()}}
    if exc:
        meta["exception"] = exc.get("exception_type")
    _set(root, "langfuse.trace.name", task)
    for k, v in meta.items():
        if v is not None:
            _set(root, f"langfuse.trace.metadata.{k}", v)
    _set(root, "langfuse.trace.tags", ["harbor", agent, *(tags or [])])
    root_id = root.span_id.hex()
    experiment_attrs = {
        "langfuse.environment": EXPERIMENT_ENVIRONMENT,
        "langfuse.experiment.id": experiment_id,
        "langfuse.experiment.name": experiment_name,
        "langfuse.experiment.dataset.id": dataset_id,
        "langfuse.experiment.item.id": item_id,
        "langfuse.experiment.item.root_observation_id": root_id,
        "langfuse.experiment.metadata.harbor_job": job_name,
        "langfuse.experiment.metadata.agent": agent,
        "langfuse.experiment.metadata.model": model,
    }
    for s in spans:  # like the SDK: every observation of the item carries the experiment attributes
        for k, v in experiment_attrs.items():
            _set(s, k, v)
    client.send_spans(rs)

    trace_id = root.trace_id.hex()
    for name, value in rewards.items():
        client.score({"id": str(uuid.uuid5(NAMESPACE, f"{trial_dir.name}\n{name}")), "traceId": trace_id,
                      "observationId": root_id, "name": name, "value": value, "environment": EXPERIMENT_ENVIRONMENT,
                      "metadata": {"harbor_trial": trial_dir.name, "harbor_task": task}})
    if exc:
        client.score({"id": str(uuid.uuid5(NAMESPACE, f"{trial_dir.name}\ntrial_error")), "traceId": trace_id,
                      "observationId": root_id, "name": "trial_error", "value": 1, "dataType": "BOOLEAN",
                      "environment": EXPERIMENT_ENVIRONMENT, "comment": f"{exc.get('exception_type')}: "
                      f"{str(exc.get('exception_message', ''))[:500]}"})
    return TrialExport(trial=trial_dir.name, task=task, dataset=dataset, item_id=item_id, experiment_id=experiment_id,
                       experiment_name=experiment_name, trace_id=trace_id, root_observation_id=root_id,
                       rewards=rewards, error=exc.get("exception_type"), agent=agent, model=model)


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


__all__ = ["TrialExport", "export_trial", "run_scores", "item_id_for", "experiment_id_for", "trial_dirs", "job_name_of"]
