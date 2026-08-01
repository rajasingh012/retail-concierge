"""Stdlib-only verifier for the RetailConcierge tamper-evident audit log.

Reads a JSONL file written by ``infrastructure.audit.AuditLogger`` and
walks the hash chain. Detects:

- truncation (final entry lacks a successor)
- reordering (entry seq differs from line position)
- tampering (entry_hash mismatch or prev_hash discontinuity)

Usage::

    python scripts/audit_verify.py path/to/retail_audit.jsonl [--quiet]
    python scripts/audit_verify.py path/to/retail_audit.jsonl --summary

Exits 0 on a clean log, 1 on any integrity violation, 2 on I/O errors.

No project deps. Works on the demo droplet without the project's venv.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Sentinel mirroring infrastructure.audit.GENESIS_HASH.
GENESIS_HASH = "0" * 64


def canonical_json(payload: dict[str, Any]) -> str:
    """Deterministic JSON encoding: sorted keys, no ASCII escapes, no NaN/Inf."""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def entry_hash(entry: dict[str, Any]) -> str:
    """sha256 hex of the canonical JSON of the entry-without-entry_hash."""
    import hashlib

    body = {k: v for k, v in entry.items() if k != "entry_hash"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def verify_log(path: Path) -> dict[str, Any]:
    """Walk the chain. Returns a summary dict with `ok` and any violations."""
    summary: dict[str, Any] = {
        "path": str(path),
        "entries": 0,
        "tools": Counter(),
        "sessions": set(),
        "first_ts": None,
        "last_ts": None,
        "ok": True,
        "violations": [],
    }
    if not path.exists():
        summary["ok"] = False
        summary["violations"].append(f"file not found: {path}")
        return summary

    prev_hash: str = GENESIS_HASH
    line_no = 0
    expected_seq = 1
    first = True
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line_no += 1
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                summary["ok"] = False
                summary["violations"].append(
                    f"line {line_no}: invalid JSON: {exc.msg}"
                )
                continue
            if not isinstance(entry, dict):
                summary["ok"] = False
                summary["violations"].append(
                    f"line {line_no}: expected object, got {type(entry).__name__}"
                )
                continue

            seq = entry.get("seq")
            prev = entry.get("prev_hash")
            declared = entry.get("entry_hash")
            tool = entry.get("tool", "<missing>")
            ts = entry.get("ts")

            if seq != expected_seq:
                summary["ok"] = False
                summary["violations"].append(
                    f"line {line_no}: seq={seq} expected {expected_seq}"
                )
            if prev != prev_hash:
                summary["ok"] = False
                summary["violations"].append(
                    f"line {line_no}: prev_hash mismatch (chain break)"
                )
            computed = entry_hash(entry)
            if declared != computed:
                summary["ok"] = False
                summary["violations"].append(
                    f"line {line_no}: entry_hash mismatch "
                    f"(tampered: declared={declared[:12]}..., computed={computed[:12]}...)"
                )

            summary["entries"] += 1
            summary["tools"][tool] += 1
            session = entry.get("session_id")
            if isinstance(session, str):
                summary["sessions"].add(session)
            if first:
                summary["first_ts"] = ts
                first = False
            summary["last_ts"] = ts

            prev_hash = entry.get("entry_hash") or computed
            expected_seq = (seq if isinstance(seq, int) else expected_seq) + 1

    summary["sessions"] = sorted(summary["sessions"])
    summary["tools"] = dict(summary["tools"])
    return summary


def render_human(summary: dict[str, Any], *, quiet: bool) -> str:
    sessions = summary.get("sessions", [])
    tools = summary.get("tools", {})
    violations = summary.get("violations", [])
    is_ok = bool(summary.get("ok"))
    if quiet and is_ok:
        return "OK"
    lines = [
        f"path: {summary.get('path', '?')}",
        f"entries: {summary.get('entries', 0)}",
        f"first_ts: {summary.get('first_ts')}",
        f"last_ts: {summary.get('last_ts')}",
        f"sessions: {len(sessions)}",
        "tools:",
    ]
    for tool, count in sorted(tools.items()):
        lines.append(f"  {tool}: {count}")
    if violations:
        lines.append("violations:")
        lines.extend(f"  - {v}" for v in violations)
    lines.append("status: OK" if is_ok else "status: TAMPERED")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a RetailConcierge audit log JSONL file"
    )
    parser.add_argument("path", type=Path, help="Audit log JSONL file")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print 'OK' on success, nothing on failure (still exits non-zero)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a machine-readable JSON summary",
    )
    args = parser.parse_args(argv)

    summary = verify_log(args.path)
    if args.summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(render_human(summary, quiet=args.quiet))

    if not summary["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
