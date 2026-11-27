"""A small Langfuse HTTP client: OTLP spans, datasets, dataset items, scores."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.parse
from collections.abc import Iterator
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
MAX_BACKOFF = 10.0
MAX_OTLP_BYTES = 3_000_000  # stay well below Langfuse's request body limit


class LangfuseError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status  # the HTTP status, if Langfuse answered


def _retry_after(exc: HTTPError) -> float | None:
    try:
        return max(0.0, float(exc.headers.get("Retry-After", "")))
    except (TypeError, ValueError, AttributeError):
        return None


def span_batches(resource_spans: ResourceSpans, max_bytes: int = MAX_OTLP_BYTES) -> Iterator[ResourceSpans]:
    """Splits a trial's spans into requests of about max_bytes; a long trajectory is too large for one."""
    for scope_spans in resource_spans.scope_spans:
        batch, size = [], 0
        for span in scope_spans.spans:
            span_size = span.ByteSize()
            if batch and size + span_size > max_bytes:
                yield ResourceSpans(resource=resource_spans.resource,
                                    scope_spans=[ScopeSpans(scope=scope_spans.scope, spans=batch)])
                batch, size = [], 0
            batch.append(span)
            size += span_size
        if batch:
            yield ResourceSpans(resource=resource_spans.resource,
                                scope_spans=[ScopeSpans(scope=scope_spans.scope, spans=batch)])


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
            last = attempt == attempts - 1
            delay = min(2.0 ** attempt, MAX_BACKOFF)
            req = Request(self.base_url + path, data=data, method=method,
                          headers={"Authorization": self._auth, **headers})
            try:
                with urlopen(req, timeout=self.timeout) as resp:
                    return resp.read()
            except HTTPError as exc:
                body = exc.read()[:300]
                if exc.code not in RETRY_STATUS or last:
                    raise LangfuseError(f"{method} {path}: HTTP {exc.code} {body!r}", status=exc.code) from exc
                delay = min(_retry_after(exc) or delay, MAX_BACKOFF)
            except (URLError, HTTPException, TimeoutError, ConnectionError) as exc:
                if last:
                    raise LangfuseError(f"{method} {path}: {exc!r}") from exc
            time.sleep(delay)
        raise AssertionError("unreachable")

    def api(self, method: str, path: str, body: dict | None = None) -> dict:
        raw = self._request(method, path, json.dumps(body).encode() if body is not None else None,
                            {"Content-Type": "application/json"})
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except ValueError as exc:  # e.g. the HTML page of a proxy or a wrong base URL
            raise LangfuseError(f"{method} {path}: response is not JSON: {raw[:200]!r}") from exc

    def send_spans(self, resource_spans: ResourceSpans) -> None:
        """OTLP/HTTP protobuf. x-langfuse-ingestion-version 4 asks for Langfuse v4 processing:
        without it Langfuse Cloud takes the older batch path, where experiment attributes
        are not turned into experiment runs."""
        for batch in span_batches(resource_spans):
            body = ExportTraceServiceRequest(resource_spans=[batch]).SerializeToString()
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
            if exc.status != 404:
                raise
        body = {"name": name, "description": description, "metadata": metadata or {}}
        try:
            return self.api("POST", "/api/public/v2/datasets", body)["id"]
        except LangfuseError as create_error:
            try:
                return self.api("GET", path)["id"]  # created concurrently
            except LangfuseError:
                raise create_error from None  # the reason the dataset could not be created

    def get_item(self, item_id: str) -> dict | None:
        try:
            return self.api("GET", "/api/public/dataset-items/" + urllib.parse.quote(item_id, safe=""))
        except LangfuseError as exc:
            if exc.status == 404:
                return None
            raise

    def upsert_item(self, item_id: str, dataset_name: str, input: object, metadata: dict) -> None:
        self.api("POST", "/api/public/dataset-items",
                 {"id": item_id, "datasetName": dataset_name, "input": input, "metadata": metadata})

    def score(self, body: dict) -> None:
        self.api("POST", "/api/public/scores", body)
