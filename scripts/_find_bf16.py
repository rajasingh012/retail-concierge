"""Locate the BF16 Gemma 4 snapshot in the HF cache.

Prints the snapshot directory containing the BF16 weights for the
requested model. Exits 0 with the path on stdout, exits 1 with an
error on stderr if not found.

Accepts BOTH layouts:
- Sharded:   model.safetensors.index.json + model-00001-of-NNNNN.safetensors
- Single:   model.safetensors (no index)

Usage:
    python _find_bf16.py [--model SUBSTRING] [--quiet]

The substring match is case-sensitive on repo_id. Defaults preserve the
historical 26B behavior so existing call sites keep working.
"""
from __future__ import annotations

import argparse
import sys


DEFAULT_MODEL = "gemma-4-26B-A4B-it"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Substring of the HF repo_id to locate (default: {DEFAULT_MODEL!r})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the diagnostic 'scanned N repos' line on stderr",
    )
    args = parser.parse_args()

    from huggingface_hub import scan_cache_dir

    needle = args.model
    cache = scan_cache_dir()
    scanned = 0
    for repo in cache.repos:
        scanned += 1
        if needle not in str(getattr(repo, "repo_id", "")):
            continue
        for rev in repo.revisions:
            for fn in rev.files:
                # Sharded checkpoint: index file marks the snapshot.
                if fn.file_name == "model.safetensors.index.json":
                    print(rev.snapshot_path)
                    return 0
                # Single-file checkpoint (e.g. google/gemma-4-12b-it):
                # no index, weights live in model.safetensors directly.
                if fn.file_name == "model.safetensors":
                    print(rev.snapshot_path)
                    return 0
    print(
        f"BF16 model matching {needle!r} not in HF cache (scanned {scanned} repos)",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
