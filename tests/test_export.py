import asyncio
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from harbor_langfuse import cli
from harbor_langfuse.client import LangfuseClient, LangfuseError, span_batches
from harbor_langfuse.export import dataset_for, experiment_id_for, export_trial, item_id_for, run_scores, trial_spans
from harbor_langfuse.plugin import LangfusePlugin

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
        (d / "agent/trajectory.json").write_text(json.dumps(TRAJECTORY if trajectory is True else trajectory))
    result = {"trial_name": name, "task_name": "hello-world/hello-world", "source": "harbor/hello-world",
              "config": {"job_id": "job-1", "agent": {"name": "terminus-2", "model_name": "anthropic/claude-sonnet-5-5"}},
              "agent_info": {"version": "2.0.0"}, "exception_info": exception,
              "verifier_result": None if exception else {"rewards": {"reward": reward}},
              "agent_execution": {"started_at": "2026-11-25T15:38:30Z", "finished_at": "2026-11-25T15:38:40Z"}}
    (d / "result.json").write_text(json.dumps(result))
    return d


@pytest.fixture()
def server():
    calls, items = [], {}

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
            if self.path.startswith("/api/public/dataset-items/"):
                item = items.get(self.path.rsplit("/", 1)[1])
                return self._reply(200, item) if item else self._reply(404, {"message": "not found"})
            self._reply(200, {"data": []})

        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            calls.append(("POST", self.path, self.headers, body))
            if self.path == "/api/public/dataset-items":
                items[json.loads(body)["id"]] = json.loads(body)
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


ERROR = {"exception_type": "AgentTimeoutError", "exception_message": "took too long"}


def posts(calls, path):
    return [json.loads(c[3]) for c in calls if c[0] == "POST" and c[1] == path]


def test_trial_without_trajectory_keeps_the_instruction_of_its_dataset_item(tmp_path, server):
    url, calls = server
    client = LangfuseClient(url, "pk", "sk")
    export_trial(client, trial(tmp_path, "hello-world__a"), job_name="my-job")
    export_trial(client, trial(tmp_path, "hello-world__b", exception=ERROR, trajectory=False), job_name="my-job")
    export_trial(client, trial(tmp_path, "hello-world__c"), job_name="my-job")
    items = posts(calls, "/api/public/dataset-items")
    assert len(items) == 1 and items[0]["input"]["instruction"] == "Create hello.txt"   # written once, never overwritten


def test_failed_trial_with_trajectory_is_marked_as_an_error(tmp_path, server):
    url, calls = server
    export_trial(LangfuseClient(url, "pk", "sk"), trial(tmp_path, exception=ERROR), job_name="my-job")
    root = next(s for s in spans_sent(calls) if not s.parent_span_id)
    assert root.status.code == 2 and root.status.message == "AgentTimeoutError: took too long"


def test_spans_without_timestamps_get_the_time_of_the_trial(tmp_path, server):
    url, calls = server
    trajectory = copy.deepcopy(TRAJECTORY)
    for step in trajectory["steps"]:
        del step["timestamp"]
    export_trial(LangfuseClient(url, "pk", "sk"), trial(tmp_path, trajectory=trajectory), job_name="my-job")
    started = 1795621110 * 10**9   # agent_execution.started_at
    assert all(started <= s.start_time_unix_nano <= s.end_time_unix_nano < started + 10**9 for s in spans_sent(calls))


def test_trajectory_without_steps_falls_back_to_a_single_span(tmp_path, server):
    url, calls = server
    export_trial(LangfuseClient(url, "pk", "sk"), trial(tmp_path, trajectory={"steps": "broken"}), job_name="my-job")
    assert len(spans_sent(calls)) == 1


def test_dataset_name_drops_the_version():
    assert dataset_for({"source": "terminal-bench/terminal-bench-2-1@1.0"}, "job") == "terminal-bench/terminal-bench-2-1"
    assert dataset_for({"source": "terminal-bench@2.0"}, "job") == "harbor/terminal-bench"
    assert dataset_for({"source": None}, "job") == "harbor/job"


def test_long_trajectories_are_sent_in_several_requests(tmp_path):
    rs = trial_spans(trial(tmp_path), {})
    spans = [s for ss in rs.scope_spans for s in ss.spans]
    batches = list(span_batches(rs, max_bytes=1))
    assert len(batches) == len(spans) > 1
    assert [s.span_id for b in batches for ss in b.scope_spans for s in ss.spans] == [s.span_id for s in spans]
    assert all(b.resource == rs.resource for b in batches)
    assert len(list(span_batches(rs))) == 1


def test_dataset_creation_error_is_not_hidden(server, monkeypatch):
    url, _ = server
    client = LangfuseClient(url, "pk", "sk")
    real = client.api

    def api(method, path, body=None):
        if method == "POST":
            raise LangfuseError("POST /api/public/v2/datasets: HTTP 400 b'invalid name'", status=400)
        return real(method, path, body)

    monkeypatch.setattr(client, "api", api)
    with pytest.raises(LangfuseError, match="invalid name"):
        client.ensure_dataset("bad")


class Event:
    def __init__(self, trial_dir):
        self.config = type("Config", (), {"trial_name": trial_dir.name})()


def plugin_for(url, job_dir, **kwargs):
    plugin = LangfusePlugin(base_url=url, public_key="pk", secret_key="sk", **kwargs)
    plugin._job_dir, plugin._job_name, plugin._job_id = job_dir, "my-job", "job-1"
    return plugin


def test_plugin_options_parsed_as_numbers_by_harbor_are_used_as_text(server):
    plugin = LangfusePlugin(base_url=server[0], public_key="pk", secret_key="sk", tags=1, run_name=2026, dataset=7)
    assert (plugin._tags, plugin._run_name, plugin._dataset) == (["1"], "2026", "7")
    assert LangfusePlugin(base_url=server[0], public_key="pk", secret_key="sk", tags="a, b,")._tags == ["a", "b"]


def test_plugin_never_fails_a_trial_over_an_export_error(tmp_path, server, monkeypatch):
    plugin = plugin_for(server[0], tmp_path)
    monkeypatch.setattr("harbor_langfuse.plugin.export_trial", lambda *a, **k: {}["unexpected"])
    asyncio.run(plugin._on_trial_ended(Event(trial(tmp_path))))   # a KeyError must not reach Harbor
    assert plugin._failures == 1


def test_resumed_job_scores_the_whole_run(tmp_path, server):
    url, calls = server
    trial(tmp_path, "t__0", reward=0.0)                       # finished before the job was resumed
    trial(tmp_path, "t__1", exception=ERROR, trajectory=False)
    plugin = plugin_for(url, tmp_path)
    asyncio.run(plugin._on_trial_ended(Event(trial(tmp_path, "t__2", reward=1.0))))
    calls.clear()
    asyncio.run(plugin.on_job_end(None))
    scores = {s["name"]: (s["value"], s["metadata"]["trials"]) for s in posts(calls, "/api/public/scores")}
    assert scores == {"mean_reward": (0.5, 3), "error_rate": (pytest.approx(1 / 3), 3)}


def test_cli_exports_the_other_trials_when_one_is_broken(tmp_path, server, monkeypatch, capsys):
    url, calls = server
    for key, value in {"LANGFUSE_BASE_URL": url, "LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}.items():
        monkeypatch.setenv(key, value)
    job = tmp_path / "job"
    trial(job, "a__1")
    (trial(job, "b__1") / "result.json").write_text("{not json")
    assert cli.main(["export", str(job)]) == 1
    assert "could not export b__1" in capsys.readouterr().err
    assert len(posts(calls, "/api/public/scores")) == 3   # reward of a__1, mean_reward and error_rate of its run
    (job / "b__1" / "result.json").write_text((job / "a__1" / "result.json").read_text())
    assert cli.main(["export", str(job)]) == 0


def test_cli_reports_missing_credentials(tmp_path, monkeypatch, capsys):
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert cli.main(["export", str(tmp_path)]) == 1
    assert "LANGFUSE_PUBLIC_KEY" in capsys.readouterr().err


def test_transient_errors_are_retried(monkeypatch):
    replies, waits = [(503, {"Retry-After": "3"}), (200, {})], []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            code, headers = replies.pop(0)
            self.send_response(code)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("content-length", "2"); self.end_headers(); self.wfile.write(b"{}")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr("harbor_langfuse.client.time.sleep", waits.append)
    try:
        assert LangfuseClient(f"http://127.0.0.1:{srv.server_address[1]}", "pk", "sk").check() == {}
    finally:
        srv.shutdown()
    assert waits == [3.0] and not replies
