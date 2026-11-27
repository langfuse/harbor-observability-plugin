import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from harbor_langfuse.client import LangfuseClient
from harbor_langfuse.export import experiment_id_for, export_trial, item_id_for, run_scores

TRAJECTORY = {
    "schema_version": "ATIF-v1.8", "session_id": "s-1",
    "agent": {"name": "terminus-2", "version": "2.0.0", "model_name": "anthropic/claude-sonnet-5-5"},
    "steps": [
        {"step_id": 1, "timestamp": "2026-11-25T15:38:30+00:00", "source": "user", "message": "Create hello.txt"},
        {"step_id": 2, "timestamp": "2026-11-25T15:38:35+00:00", "source": "agent", "model_name": "claude-sonnet-5-5",
         "message": "Writing the file.", "tool_calls": [{"tool_call_id": "c1", "function_name": "bash_command",
                                                         "arguments": {"keystrokes": "echo hi > hello.txt\n"}}],
         "observation": {"results": [{"source_call_id": "c1", "content": "ok"}]},
         "metrics": {"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 400, "cost_usd": 0.004}},
        {"step_id": 3, "timestamp": "2026-11-25T15:38:37+00:00", "source": "agent", "model_name": "claude-sonnet-5-5",
         "message": "Done.", "metrics": {"prompt_tokens": 1200, "completion_tokens": 50, "cost_usd": 0.002}},
    ],
    "final_metrics": {"total_prompt_tokens": 2200, "total_completion_tokens": 150, "total_cost_usd": 0.006},
}


def trial(tmp: Path, name="hello-world__abc", reward=1.0, exception=None, trajectory=True) -> Path:
    d = tmp / name
    (d / "agent").mkdir(parents=True)
    if trajectory:
        (d / "agent/trajectory.json").write_text(json.dumps(TRAJECTORY))
    result = {"trial_name": name, "task_name": "hello-world/hello-world", "source": "harbor/hello-world",
              "config": {"job_id": "job-1", "agent": {"name": "terminus-2", "model_name": "anthropic/claude-sonnet-5-5"}},
              "agent_info": {"version": "2.0.0"}, "exception_info": exception,
              "verifier_result": None if exception else {"rewards": {"reward": reward}},
              "agent_execution": {"started_at": "2026-11-25T15:38:30Z", "finished_at": "2026-11-25T15:38:40Z"}}
    (d / "result.json").write_text(json.dumps(result))
    return d


@pytest.fixture()
def server():
    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            calls.append(("GET", self.path, self.headers, None))
            if self.path.startswith("/api/public/v2/datasets/"):
                return self._reply(404, {"message": "not found"})
            self._reply(200, {"data": []})

        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            calls.append(("POST", self.path, self.headers, body))
            self._reply(200, {"id": "ds-1"} if self.path == "/api/public/v2/datasets" else {})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", calls
    srv.shutdown()


def spans_sent(calls):
    req = ExportTraceServiceRequest()
    req.ParseFromString(next(c[3] for c in calls if c[1] == "/api/public/otel/v1/traces"))
    return [s for rs in req.resource_spans for ss in rs.scope_spans for s in ss.spans]


def attrs(span):
    out = {}
    for a in span.attributes:
        v = a.value
        out[a.key] = v.string_value or v.double_value or [x.string_value for x in v.array_value.values] or v.bool_value
    return out


def test_trial_becomes_an_experiment_item_with_linked_trace_and_scores(tmp_path, server):
    url, calls = server
    client = LangfuseClient(url, "pk", "sk")
    ex = export_trial(client, trial(tmp_path), job_name="my-job")
    otlp = next(c for c in calls if c[1] == "/api/public/otel/v1/traces")
    assert otlp[2].get("x-langfuse-ingestion-version") == "4"   # v4 processing: experiments are built from the spans
    spans = spans_sent(calls)
    root = next(s for s in spans if not s.parent_span_id)
    ra = attrs(root)
    assert ra["langfuse.trace.name"] == "hello-world/hello-world"
    assert ra["langfuse.trace.metadata.harbor_trial"] == "hello-world__abc"
    assert ra["langfuse.trace.metadata.reward_reward"] == 1.0
    assert not [k for k in ra if k.startswith("llm.token_count") or k.startswith("llm.cost")]  # no double counting
    llm = [attrs(s) for s in spans if attrs(s).get("openinference.span.kind") == "LLM"]
    assert llm and all("llm.token_count.prompt" in a for a in llm)
    for s in spans:
        a = attrs(s)
        assert a["langfuse.environment"] == "sdk-experiment"
        assert a["langfuse.experiment.id"] == experiment_id_for("job-1", "terminus-2", "anthropic/claude-sonnet-5-5", "harbor/hello-world")
        assert a["langfuse.experiment.item.id"] == item_id_for("harbor/hello-world", "hello-world/hello-world")
        assert a["langfuse.experiment.item.root_observation_id"] == root.span_id.hex()
        assert a["langfuse.experiment.dataset.id"] == "ds-1"
    item = json.loads(next(c[3] for c in calls if c[1] == "/api/public/dataset-items"))
    assert item["datasetName"] == "harbor/hello-world" and item["input"]["instruction"] == "Create hello.txt"
    score = json.loads(next(c[3] for c in calls if c[1] == "/api/public/scores"))
    assert (score["traceId"], score["observationId"], score["name"], score["value"]) == (ex.trace_id, ex.root_observation_id, "reward", 1.0)
    assert ex.experiment_name == "my-job (terminus-2, claude-sonnet-5-5)"


def test_failed_trial_without_trajectory_still_lands_in_the_run(tmp_path, server):
    url, calls = server
    ex = export_trial(LangfuseClient(url, "pk", "sk"), trial(tmp_path, exception={"exception_type": "AgentTimeoutError",
                      "exception_message": "took too long"}, trajectory=False), job_name="my-job")
    spans = spans_sent(calls)
    assert len(spans) == 1 and spans[0].status.code == 2   # ERROR
    assert attrs(spans[0])["langfuse.experiment.item.id"] == ex.item_id
    scores = [json.loads(c[3]) for c in calls if c[1] == "/api/public/scores"]
    assert [s["name"] for s in scores] == ["trial_error"]


def test_run_scores_average_rewards_per_run(tmp_path, server):
    url, calls = server
    client = LangfuseClient(url, "pk", "sk")
    exports = [export_trial(client, trial(tmp_path, f"t__{i}", reward=r), job_name="j") for i, r in enumerate([1.0, 0.0, 1.0, 1.0])]
    calls.clear()
    bodies = run_scores(client, exports)
    by = {b["name"]: b["value"] for b in bodies}
    assert by == {"mean_reward": 0.75, "error_rate": 0.0}
    assert {b["datasetRunId"] for b in bodies} == {exports[0].experiment_id}


def test_ids_are_stable_across_jobs():
    assert item_id_for("harbor/hello-world", "t") == item_id_for("harbor/hello-world", "t")
    assert experiment_id_for("job-1", "a", "m", "d") != experiment_id_for("job-2", "a", "m", "d")
    assert experiment_id_for("job-1", "a", "m", "d1") != experiment_id_for("job-1", "a", "m", "d2")
    assert len(experiment_id_for("job-1", "a", "m", "d")) == 16


def test_item_instruction_comes_from_the_task_not_the_trajectory(tmp_path, server):
    # Codex, for example, sends an <environment_context> block as its first user message.
    from harbor.models.trial.config import TaskConfig, TrialConfig

    task_dir = tmp_path / "tasks" / "hello-world"
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Create hello.txt with the text hi.\n")
    d = trial(tmp_path / "jobs")
    traj = json.loads((d / "agent/trajectory.json").read_text())
    traj["steps"][0]["message"] = "<environment_context><cwd>/app</cwd></environment_context>"
    (d / "agent/trajectory.json").write_text(json.dumps(traj))
    (d / "config.json").write_text(TrialConfig(task=TaskConfig(path=task_dir), trial_name=d.name).model_dump_json())
    url, calls = server
    export_trial(LangfuseClient(url, "pk", "sk"), d, job_name="my-job")
    item = json.loads(next(c[3] for c in calls if c[1] == "/api/public/dataset-items"))
    assert item["input"]["instruction"] == "Create hello.txt with the text hi."
