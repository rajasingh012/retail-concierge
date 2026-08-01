"""Tamper-evident audit log for catalog and finalizer tool calls.

Each entry is a single JSONL line with the schema:
    {"seq": N, "ts": "ISO8601", "session_id": "...", "tool": "...",
     "args": {...}, "result_meta": {...}, "prev_hash": "...",
     "entry_hash": "<sha256 of canonical_json(entry without entry_hash)>"}

The chain head is ``prev_hash`` of the most-recent entry; verification
re-hashes every line and confirms the prev_hash of line N matches the
entry_hash of line N-1.

Design choices:
- stdlib only (json + hashlib). No project deps so the verifier ships
  as a single script and works on the demo droplet without venv.
- canonical_json is deterministic (sorted keys, UTF-8 NFC, no ASCII
  escaping) so the same logical entry always hashes the same.
- append-only JSONL. Existing line edits break the chain by changing
  the canonical_hash; replays of a valid log produce identical hashes.
- thread-safe. A lock serializes writes; readers do not block writers.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infrastructure.agent_tools import _CATALOG_NAMESPACE  # type: ignore[import-not-found]


SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64  # sha256 hex of empty bytes — sentinel for seq=0


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialize a dict deterministically for hashing.

    Sorted keys + UTF-8 + no ASCII escaping + reject NaN/Inf (invalid in
    canonical JSON). Equal dicts always hash equal; key order does not
    matter.
    """
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash_entry(entry: dict[str, Any]) -> str:
    """sha256 hex of the canonical JSON of the entry-without-entry_hash."""
    body = {k: v for k, v in entry.items() if k != "entry_hash"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class AuditLogger:
    """Append-only JSONL audit log with hash-chained integrity.

    Writes one entry per tool invocation. Safe to share between threads;
    the underlying file is opened per-append (mode="a") so concurrent
    processes append safely on POSIX, and the lock guarantees in-process
    writers see a coherent chain head.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        session_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        self._path = Path(path)
        self._session_id = session_id or str(uuid.uuid4())
        self._enabled = enabled
        self._lock = threading.Lock()
        self._head_hash = GENESIS_HASH
        self._next_seq = self._load_next_seq()
        if enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def session_id(self) -> str:
        return self._session_id

    def enable(self) -> None:
        self._enabled = True
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def disable(self) -> None:
        self._enabled = False

    def _load_next_seq(self) -> int:
        """Read the tail of the log on open to learn the current chain head."""
        if not self._path.exists():
            return 1
        last_hash: str | None = None
        last_seq = 0
        with self._path.open("rb") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    # Corrupt tail — refuse to append past it.
                    raise ValueError(
                        f"audit log {self._path} has a corrupt tail; "
                        "verify or rotate before appending"
                    )
                last_hash = entry.get("entry_hash") or entry.get("prev_hash")
                last_seq = max(last_seq, int(entry.get("seq", 0)))
        self._head_hash = last_hash or GENESIS_HASH
        return last_seq + 1

    def record(
        self,
        tool: str,
        args: dict[str, Any],
        result_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Append one audit entry. Returns the entry written, or None when disabled."""
        if not self._enabled:
            return None
        meta = dict(result_meta or {})
        meta.setdefault("namespace", _CATALOG_NAMESPACE)
        with self._lock:
            seq = self._next_seq
            entry: dict[str, Any] = {
                "seq": seq,
                "ts": now_iso(),
                "session_id": self._session_id,
                "tool": tool,
                "args": args,
                "result_meta": meta,
                "schema": SCHEMA_VERSION,
                "prev_hash": self._head_hash,
            }
            entry["entry_hash"] = _hash_entry(entry)
            line = canonical_json(entry) + "\n"
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            self._head_hash = entry["entry_hash"]
            self._next_seq = seq + 1
            return entry
