"""Usage metrics and request-forwarding record store.

Collects a per-request forwarding record (endpoint, model, stream, status,
latency, token usage) and maintains cumulative aggregates for the Admin
dashboard. Records are persisted to a JSONL append log and aggregates to a
small sidecar summary so totals survive process restarts.
"""

import atexit
import json
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from loguru import logger

MAX_RECORDS = 5000
DEFAULT_LOG_PATH = "logs/usage.jsonl"
DEFAULT_SUMMARY_PATH = "logs/usage_summary.json"
SUMMARY_FLUSH_INTERVAL_S = 5.0


def _empty_model_agg() -> dict[str, int]:
    return {
        "requests": 0,
        "success": 0,
        "failed": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


class MetricsStore:
    """Thread-safe in-memory + on-disk store for forwarding metrics."""

    def __init__(
        self,
        log_path: str = DEFAULT_LOG_PATH,
        summary_path: str = DEFAULT_SUMMARY_PATH,
    ) -> None:
        self._log_path = Path(log_path)
        self._summary_path = Path(summary_path)
        self._lock = threading.Lock()

        self._records: deque[dict[str, Any]] = deque(maxlen=MAX_RECORDS)

        self.total_requests = 0
        self.total_success = 0
        self.total_failed = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        self.by_model: dict[str, dict[str, int]] = defaultdict(_empty_model_agg)
        self.by_endpoint: dict[str, dict[str, int]] = defaultdict(_empty_model_agg)

        self._write_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._summary_dirty = False
        self._last_summary_flush = time.monotonic()

        self._load_summary()
        self._load_recent_records()

        # Disk writes run on a daemon thread so request-path record() calls
        # never block the async event loop on file I/O.
        self._writer = threading.Thread(
            target=self._writer_loop, name="metrics-writer", daemon=True
        )
        self._writer.start()
        atexit.register(self.flush)

    # ── persistence ──────────────────────────────────────────────────────

    def _load_summary(self) -> None:
        if not self._summary_path.exists():
            return
        try:
            data = json.loads(self._summary_path.read_text(encoding="utf-8"))
            self.total_requests = int(data.get("total_requests", 0))
            self.total_success = int(data.get("total_success", 0))
            self.total_failed = int(data.get("total_failed", 0))
            self.total_input_tokens = int(data.get("total_input_tokens", 0))
            self.total_output_tokens = int(data.get("total_output_tokens", 0))
            for model, agg in (data.get("by_model") or {}).items():
                self.by_model[model] = _empty_model_agg()
                self.by_model[model].update({k: int(v) for k, v in agg.items() if isinstance(v, int)})
            for endpoint, agg in (data.get("by_endpoint") or {}).items():
                self.by_endpoint[endpoint] = _empty_model_agg()
                self.by_endpoint[endpoint].update({k: int(v) for k, v in agg.items() if isinstance(v, int)})
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(f"Failed to load usage summary from {self._summary_path}: {exc}")

    def _load_recent_records(self) -> None:
        if not self._log_path.exists():
            return
        try:
            lines = self._log_path.read_text(encoding="utf-8").splitlines()
            # Keep only the most recent MAX_RECORDS entries.
            for line in lines[-MAX_RECORDS:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    logger.warning(f"Skipping malformed usage record line in {self._log_path}")
                    continue
                self._records.append(rec)
        except OSError as exc:
            logger.warning(f"Failed to load usage records from {self._log_path}: {exc}")

    def _persist_record(self, rec: dict[str, Any]) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning(f"Failed to append usage record: {exc}")

    def _persist_summary(self) -> None:
        try:
            self._summary_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "total_requests": self.total_requests,
                "total_success": self.total_success,
                "total_failed": self.total_failed,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "by_model": dict(self.by_model),
                "by_endpoint": dict(self.by_endpoint),
            }
            tmp_path = self._summary_path.with_suffix(self._summary_path.suffix + ".tmp")
            # Write-then-rename so a crash mid-write cannot corrupt the summary.
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(self._summary_path)
        except OSError as exc:
            logger.warning(f"Failed to write usage summary: {exc}")

    def _writer_loop(self) -> None:
        """Drain queued records to the JSONL log and flush the summary."""
        while True:
            rec = self._write_queue.get()
            try:
                if rec is None:  # shutdown sentinel
                    return
                self._persist_record(rec)
                now = time.monotonic()
                if self._summary_dirty and (
                    now - self._last_summary_flush >= SUMMARY_FLUSH_INTERVAL_S
                ):
                    with self._lock:
                        self._persist_summary()
                    self._summary_dirty = False
                    self._last_summary_flush = now
            finally:
                self._write_queue.task_done()

    def flush(self) -> None:
        """Block until all queued records are on disk and summary is current."""
        self._write_queue.join()
        with self._lock:
            if self._summary_dirty:
                self._persist_summary()
                self._summary_dirty = False
                self._last_summary_flush = time.monotonic()

    # ── recording ────────────────────────────────────────────────────────

    def record(
        self,
        *,
        endpoint: str,
        model: str,
        stream: bool,
        status: str,
        status_code: int,
        latency_ms: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: str | None = None,
    ) -> None:
        rec: dict[str, Any] = {
            "id": f"req_{uuid.uuid4().hex[:12]}",
            "ts": time.time(),
            "endpoint": endpoint,
            "model": model or "unknown",
            "stream": bool(stream),
            "status": status,
            "status_code": int(status_code),
            "latency_ms": int(latency_ms),
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "error": error,
        }

        with self._lock:
            self._records.append(rec)
            self.total_requests += 1
            if status == "success":
                self.total_success += 1
            else:
                self.total_failed += 1
            self.total_input_tokens += rec["input_tokens"]
            self.total_output_tokens += rec["output_tokens"]

            ma = self.by_model[rec["model"]]
            ma["requests"] += 1
            ma["success"] += 1 if status == "success" else 0
            ma["failed"] += 1 if status != "success" else 0
            ma["input_tokens"] += rec["input_tokens"]
            ma["output_tokens"] += rec["output_tokens"]

            ea = self.by_endpoint[rec["endpoint"]]
            ea["requests"] += 1
            ea["success"] += 1 if status == "success" else 0
            ea["failed"] += 1 if status != "success" else 0
            ea["input_tokens"] += rec["input_tokens"]
            ea["output_tokens"] += rec["output_tokens"]

            self._summary_dirty = True
            # Hand disk I/O to the background writer thread (non-blocking).
            self._write_queue.put(rec)

    # ── queries ──────────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "totals": {
                    "requests": self.total_requests,
                    "success": self.total_success,
                    "failed": self.total_failed,
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                    "total_tokens": self.total_input_tokens + self.total_output_tokens,
                },
                "by_model": [
                    {"model": m, **dict(agg)}
                    for m, agg in sorted(self.by_model.items(), key=lambda kv: -kv[1]["requests"])
                ],
                "by_endpoint": [
                    {"endpoint": e, **dict(agg)}
                    for e, agg in sorted(self.by_endpoint.items(), key=lambda kv: -kv[1]["requests"])
                ],
            }

    def records(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        status: str | None = None,
        endpoint: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            # Newest records first: reverse before filtering/paginating.
            items = list(reversed(self._records))
            if status:
                items = [r for r in items if r["status"] == status]
            if endpoint:
                items = [r for r in items if r["endpoint"] == endpoint]
            if model:
                items = [r for r in items if r["model"] == model]

            total = len(items)
            start = (page - 1) * page_size
            end = start + page_size
            page_items = items[start:end]

            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "pages": (total + page_size - 1) // page_size if page_size > 0 else 0,
                "records": page_items,
            }

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self.total_requests = 0
            self.total_success = 0
            self.total_failed = 0
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.by_model.clear()
            self.by_endpoint.clear()
            self._summary_dirty = False
            # Discard records queued before the reset so the writer thread
            # cannot re-append them after the log file is removed.
            while True:
                try:
                    self._write_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._write_queue.task_done()
            for path in (self._log_path, self._summary_path):
                try:
                    if path.exists():
                        path.unlink()
                except OSError as exc:
                    logger.warning(f"Failed to remove {path}: {exc}")


_default_store: MetricsStore | None = None
_default_store_lock = threading.Lock()


def get_metrics_store() -> MetricsStore:
    """Lazily create the process-wide store on first use.

    Deferred construction keeps ``import nim_proxy.core.metrics`` free of
    filesystem side effects and lets tests substitute a tmp-path store.
    """
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = MetricsStore()
    return _default_store


def _set_metrics_store(store: MetricsStore | None) -> None:
    """Replace the process-wide store (test hook)."""
    global _default_store
    with _default_store_lock:
        _default_store = store
