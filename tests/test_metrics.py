"""Tests for usage metrics store and admin usage endpoints."""

import pytest
from fastapi.testclient import TestClient

from nim_proxy.api.app import app
from nim_proxy.config.settings import settings
from nim_proxy.core.metrics import MetricsStore

client = TestClient(app)


@pytest.fixture(autouse=True)
def _no_auth(monkeypatch):
    monkeypatch.setattr(settings, "proxy_api_token", "")


@pytest.fixture(autouse=True)
def _isolated_store(monkeypatch, tmp_path):
    """Point the process-wide metrics store at a tmp path so endpoint tests
    never touch the real ./logs directory."""
    import nim_proxy.core.metrics as metrics_mod

    store = MetricsStore(
        str(tmp_path / "endpoint_usage.jsonl"),
        str(tmp_path / "endpoint_usage_summary.json"),
    )
    monkeypatch.setattr(metrics_mod, "_default_store", store)


# ── MetricsStore unit tests ──────────────────────────────────────────────


def _make_store(tmp_path) -> MetricsStore:
    return MetricsStore(
        str(tmp_path / "usage.jsonl"),
        str(tmp_path / "usage_summary.json"),
    )


def test_record_and_summary_aggregation(tmp_path):
    store = _make_store(tmp_path)
    store.record(
        endpoint="/v1/messages", model="m1", stream=False,
        status="success", status_code=200, latency_ms=5,
        input_tokens=10, output_tokens=20,
    )
    store.record(
        endpoint="/v1/messages", model="m1", stream=True,
        status="error", status_code=500, latency_ms=7,
        input_tokens=3, output_tokens=0, error="boom",
    )
    store.record(
        endpoint="/v1/responses", model="m2", stream=False,
        status="success", status_code=200, latency_ms=9,
        input_tokens=5, output_tokens=15,
    )

    summary = store.summary()
    totals = summary["totals"]
    assert totals["requests"] == 3
    assert totals["success"] == 2
    assert totals["failed"] == 1
    assert totals["input_tokens"] == 18
    assert totals["output_tokens"] == 35
    assert totals["total_tokens"] == 53

    by_model = {m["model"]: m for m in summary["by_model"]}
    assert by_model["m1"]["requests"] == 2
    assert by_model["m1"]["success"] == 1
    assert by_model["m1"]["failed"] == 1
    assert by_model["m2"]["requests"] == 1
    assert by_model["m2"]["output_tokens"] == 15

    by_endpoint = {e["endpoint"]: e for e in summary["by_endpoint"]}
    assert by_endpoint["/v1/messages"]["requests"] == 2
    assert by_endpoint["/v1/responses"]["requests"] == 1


def test_records_pagination_and_filters(tmp_path):
    store = _make_store(tmp_path)
    for i in range(5):
        store.record(
            endpoint="/v1/messages", model=f"m{i % 2}", stream=False,
            status="success" if i % 2 == 0 else "error",
            status_code=200 if i % 2 == 0 else 500, latency_ms=i,
        )

    page = store.records(page=1, page_size=2)
    assert page["total"] == 5
    assert page["pages"] == 3
    assert len(page["records"]) == 2
    # Newest records come first.
    assert page["records"][0]["latency_ms"] == 4
    assert page["records"][1]["latency_ms"] == 3

    success_only = store.records(page=1, page_size=10, status="success")
    assert success_only["total"] == 3
    assert all(r["status"] == "success" for r in success_only["records"])

    model_filtered = store.records(page=1, page_size=10, model="m0")
    assert model_filtered["total"] == 3
    assert all(r["model"] == "m0" for r in model_filtered["records"])

    endpoint_filtered = store.records(page=1, page_size=10, endpoint="/v1/messages")
    assert endpoint_filtered["total"] == 5


def test_persistence_reloads_summary_and_records(tmp_path):
    log = str(tmp_path / "usage.jsonl")
    summ = str(tmp_path / "usage_summary.json")
    store = MetricsStore(log, summ)
    store.record(
        endpoint="/v1/messages", model="m1", stream=False,
        status="success", status_code=200, latency_ms=5,
        input_tokens=100, output_tokens=50,
    )
    store.flush()

    reloaded = MetricsStore(log, summ)
    assert reloaded.total_requests == 1
    assert reloaded.total_input_tokens == 100
    assert reloaded.total_output_tokens == 50
    assert reloaded.records(page=1, page_size=10)["total"] == 1


def test_reset_clears_everything(tmp_path):
    store = _make_store(tmp_path)
    store.record(
        endpoint="/v1/messages", model="m1", stream=False,
        status="success", status_code=200, latency_ms=5,
        input_tokens=10, output_tokens=20,
    )
    store.flush()
    store.reset()

    summary = store.summary()
    assert summary["totals"]["requests"] == 0
    assert summary["totals"]["total_tokens"] == 0
    assert summary["by_model"] == []
    assert summary["by_endpoint"] == []
    assert store.records(page=1, page_size=10)["total"] == 0


# ── SSE usage capture ────────────────────────────────────────────────────


def test_capture_usage_from_message_start_and_delta():
    from nim_proxy.api.routes import _capture_usage_from_sse

    usage: dict = {}
    pending = _capture_usage_from_sse(
        "event: message_start\n"
        'data: {"type": "message_start", "message": {"usage": {"input_tokens": 42, "output_tokens": 1}}}\n'
        "\n",
        usage,
    )
    assert pending == ""
    assert usage == {"input_tokens": 42, "output_tokens": 1}

    pending = _capture_usage_from_sse(
        "event: message_delta\n"
        'data: {"type": "message_delta", "usage": {"output_tokens": 17}}\n'
        "\n",
        usage,
    )
    assert usage == {"input_tokens": 42, "output_tokens": 17}


def test_capture_usage_handles_events_split_across_chunks():
    from nim_proxy.api.routes import _capture_usage_from_sse

    usage: dict = {}
    # Event split mid-payload across two chunks: nothing parsed until the
    # blank-line terminator arrives.
    pending = _capture_usage_from_sse(
        'event: message_delta\ndata: {"type": "message_delta", "usa', usage
    )
    assert usage == {}
    pending = _capture_usage_from_sse('ge": {"output_tokens": 9}}\n\n', usage, pending)
    assert pending == ""
    assert usage == {"output_tokens": 9}


# ── Admin usage endpoints ────────────────────────────────────────────────


def test_usage_summary_endpoint():
    response = client.get("/admin/api/usage/summary")
    assert response.status_code == 200
    data = response.json()
    assert "totals" in data
    assert "by_model" in data
    assert "by_endpoint" in data
    assert "requests" in data["totals"]


def test_usage_records_endpoint():
    response = client.get("/admin/api/usage/records?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert "records" in data
    assert "total" in data
    assert "pages" in data
    assert "page" in data


def test_usage_reset_endpoint():
    response = client.post("/admin/api/usage/reset")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
