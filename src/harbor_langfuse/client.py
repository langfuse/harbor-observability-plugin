"""A small Langfuse HTTP client: OTLP spans, datasets, dataset items, scores."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.parse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class LangfuseError(RuntimeError):
    pass


class LangfuseClient:
    """Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL unless given."""

    def __init__(self, base_url: str | None = None, public_key: str | None = None,
                 secret_key: str | None = None, timeout: float = 30.0):
        self.base_url = (base_url or os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
                         or "https://cloud.langfuse.com").rstrip("/")
        public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        if not public_key or not secret_key:
            raise LangfuseError("set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (and LANGFUSE_BASE_URL)")
        self._auth = "Basic " + base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self.timeout = timeout

    # -- transport
    def _request(self, method: str, path: str, data: bytes | None, headers: dict, attempts: int = 4) -> bytes:
        for attempt in range(attempts):
            req = Request(self.base_url + path, data=data, method=method,
                          headers={"Authorization": self._auth, **headers})
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    return resp.read()
            except HTTPError as exc:
                body = exc.read()[:300]
                if exc.code not in RETRY_STATUS or attempt == attempts - 1:
                    raise LangfuseError(f"{method} {path}: HTTP {exc.code} {body!r}") from exc
            except (URLError, TimeoutError, ConnectionError) as exc:
                if attempt == attempts - 1:
                    raise LangfuseError(f"{method} {path}: {exc}") from exc
            time.sleep(min(2 ** attempt, 10))
        raise AssertionError("unreachable")

    def api(self, method: str, path: str, body: dict | None = None) -> dict:
        raw = self._request(method, path, json.dumps(body).encode() if body is not None else None,
                            {"Content-Type": "application/json"})
        return json.loads(raw) if raw else {}

    def send_spans(self, resource_spans: ResourceSpans) -> None:
        """OTLP/HTTP protobuf. x-langfuse-ingestion-version 4 asks for Langfuse v4 processing:
        without it Langfuse Cloud takes the older batch path, where experiment attributes
        are not turned into experiment runs."""
        body = ExportTraceServiceRequest(resource_spans=[resource_spans]).SerializeToString()
        self._request("POST", "/api/public/otel/v1/traces", body,
                      {"Content-Type": "application/x-protobuf", "x-langfuse-ingestion-version": "4"})

    # -- resources
    def check(self) -> dict:
        return self.api("GET", "/api/public/projects")

    def ensure_dataset(self, name: str, description: str | None = None, metadata: dict | None = None) -> str:
        path = "/api/public/v2/datasets/" + urllib.parse.quote(name, safe="")
        try:
            return self.api("GET", path)["id"]
        except LangfuseError as exc:
            if "HTTP 404" not in str(exc):
                raise
        body = {"name": name, "description": description, "metadata": metadata or {}}
        try:
            return self.api("POST", "/api/public/v2/datasets", body)["id"]
        except LangfuseError:
            return self.api("GET", path)["id"]  # created concurrently

    def upsert_item(self, item_id: str, dataset_name: str, input: object, metadata: dict) -> None:
        self.api("POST", "/api/public/dataset-items",
                 {"id": item_id, "datasetName": dataset_name, "input": input, "metadata": metadata})

    def score(self, body: dict) -> None:
        self.api("POST", "/api/public/scores", body)
