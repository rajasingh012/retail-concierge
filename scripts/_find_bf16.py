"""Locate the BF16 Gemma 4 26B A4B-it snapshot in the HF cache."""
import json
from huggingface_hub import scan_cache_dir

cache = scan_cache_dir()
for repo in cache.repos:
    if "gemma-4-26B-A4B-it" not in str(getattr(repo, "repo_id", "")):
        continue
    for rev in repo.revisions:
        for fn in rev.files:
            if fn.file_name == "model.safetensors.index.json":
                print(rev.snapshot_path)
                raise SystemExit(0)
raise SystemExit("BF16 Gemma 4 26B A4B-it not in HF cache")
