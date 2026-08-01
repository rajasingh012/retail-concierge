"""Tests for the tamper-evident audit log + stdlib verifier.

Covers:
- canonical_json determinism (key order independent, unicode-stable)
- chain integrity on a fresh log
- tamper detection: line edit, line delete, line reorder
- disabled logger writes nothing
- thread-safe concurrent writes maintain a coherent chain
- scripts/audit_verify.py round-trip exits 0 on clean log, 1 on tamper
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from infrastructure.audit import (
    GENESIS_HASH,
    AuditLogger,
    canonical_json,
    now_iso,
)


def test_canonical_json_is_key_order_independent() -> None:
    a = canonical_json({"tool": "search_catalog", "args": {"q": "x"}})
    b = canonical_json({"args": {"q": "x"}, "tool": "search_catalog"})
    assert a == b
    assert " " not in a  # tight separators


def test_canonical_json_preserves_unicode() -> None:
    s = canonical_json({"query": "café résumé naïve"})
    assert "café" in s
    assert "\\u00e9" not in s


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_timestamp_format_is_iso8601_utc_ms() -> None:
    ts = now_iso()
    # YYYY-MM-DDTHH:MM:SS.sss+00:00 or with 'Z'
    assert "T" in ts
    assert ts.endswith("+00:00") or ts.endswith("Z")


def test_disabled_logger_writes_nothing(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path, enabled=False)
    result = logger.record("search_catalog", {"q": "x"}, {"n": 1})
    assert result is None
    assert not log_path.exists()


def test_chain_continuity(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path)
    e1 = logger.record("find_brands", {"q": "logitech"}, {"count": 3})
    e2 = logger.record("search_catalog", {"q": "mouse"}, {"count": 12})
    assert e1 is not None and e2 is not None
    assert e1["seq"] == 1
    assert e2["seq"] == 2
    assert e1["prev_hash"] == GENESIS_HASH
    assert e2["prev_hash"] == e1["entry_hash"]
    # Each entry's entry_hash matches the canonical hash of its body
    body1 = {k: v for k, v in e1.items() if k != "entry_hash"}
    body2 = {k: v for k, v in e2.items() if k != "entry_hash"}
    import hashlib

    assert e1["entry_hash"] == hashlib.sha256(
        canonical_json(body1).encode("utf-8")
    ).hexdigest()
    assert e2["entry_hash"] == hashlib.sha256(
        canonical_json(body2).encode("utf-8")
    ).hexdigest()


def test_reopen_resumes_chain(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    session_id = "deterministic-session-id"
    a = AuditLogger(log_path, session_id=session_id)
    e1 = a.record("search_catalog", {"q": "x"}, {"count": 1})
    b = AuditLogger(log_path, session_id=session_id)
    e2 = b.record("search_catalog", {"q": "y"}, {"count": 2})
    assert e1 is not None and e2 is not None
    assert e2["seq"] == 2
    assert e2["prev_hash"] == e1["entry_hash"]


def test_reopen_on_corrupt_tail_raises(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    log_path.write_text('{"seq": 1, "garbage"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt tail"):
        AuditLogger(log_path)


def test_concurrent_writes_preserve_chain(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path)

    def worker(idx: int) -> None:
        for i in range(20):
            logger.record("search_catalog", {"thread": idx, "i": i}, {"n": i})

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 80 entries total, seq 1..80, prev_hash chain intact.
    import hashlib

    entries = []
    with log_path.open("rb") as handle:
        for raw in handle:
            stripped = raw.strip()
            if stripped:
                entries.append(json.loads(stripped))
    assert len(entries) == 80
    assert [e["seq"] for e in entries] == list(range(1, 81))
    prev = GENESIS_HASH
    for entry in entries:
        assert entry["prev_hash"] == prev
        body = {k: v for k, v in entry.items() if k != "entry_hash"}
        expected = hashlib.sha256(
            canonical_json(body).encode("utf-8")
        ).hexdigest()
        assert entry["entry_hash"] == expected
        prev = entry["entry_hash"]


def _write_clean_log(path: Path) -> None:
    logger = AuditLogger(path)
    logger.record("find_product_types", {"query": "headphone"}, {"count": 4})
    logger.record("find_brands", {"query": "sony"}, {"count": 3})
    logger.record("search_catalog", {"query": "headphone"}, {"count": 50})
    logger.record(
        "finalize_recommendations",
        {"proposed_item_ids": ["A1"]},
        {"accepted_item_ids": ["A1"], "provenance_blocked": [], "result_count": 1},
    )


def test_verifier_accepts_clean_log(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    _write_clean_log(log_path)
    result = subprocess.run(
        [sys.executable, "scripts/audit_verify.py", str(log_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_verifier_flags_tampered_line(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    _write_clean_log(log_path)
    raw = log_path.read_text(encoding="utf-8").splitlines()
    # Mutate the args of the 2nd line so the entry_hash no longer matches.
    second = json.loads(raw[1])
    second["args"]["query"] = "TAMPERED"
    raw[1] = json.dumps(second, ensure_ascii=False)
    log_path.write_text("\n".join(raw) + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/audit_verify.py", str(log_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 1
    assert "TAMPERED" in result.stdout or "entry_hash mismatch" in result.stdout


def test_verifier_flags_missing_line(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    _write_clean_log(log_path)
    raw = log_path.read_text(encoding="utf-8").splitlines()
    raw.pop(1)  # delete the 2nd entry
    log_path.write_text("\n".join(raw) + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/audit_verify.py", str(log_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 1


def test_verifier_flags_reorder(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    _write_clean_log(log_path)
    raw = log_path.read_text(encoding="utf-8").splitlines()
    raw[0], raw[2] = raw[2], raw[0]  # swap lines 0 and 2
    log_path.write_text("\n".join(raw) + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/audit_verify.py", str(log_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 1


def test_verifier_missing_file(tmp_path: Path) -> None:
    log_path = tmp_path / "missing.jsonl"
    result = subprocess.run(
        [sys.executable, "scripts/audit_verify.py", str(log_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 1
    assert "not found" in result.stdout
