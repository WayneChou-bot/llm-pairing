"""Catalog snapshot builder (T-005). RUNS ON THE OWNER'S MACHINE (network).

Fetches top GGUF model repos from the Hugging Face API, resolves each to
its base model's config.json + parameter count, normalizes through
llmpairing.catalog.mapper, and emits a VERSIONED SNAPSHOT:

    catalog/catalog-<YYYYMMDD>.json        the snapshot
    catalog/catalog-<YYYYMMDD>.sha256      its hash (reports must cite it)
    catalog_cache/...                      every raw API response (provenance)

Reproducibility over freshness (F-11): consumers load a snapshot by version;
nothing queries the network at match time.

Usage (from the repo root, conda env active):
    python tools\\catalog\\build_catalog.py                # top 25 by downloads
    python tools\\catalog\\build_catalog.py --top 40
    python tools\\catalog\\build_catalog.py --repos unsloth/gemma-3-12b-it-GGUF,Qwen/Qwen3-30B-A3B-GGUF
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from llmpairing.catalog import mapper  # noqa: E402

HF = "https://huggingface.co"
CACHE = REPO / "catalog_cache"
OUT_DIR = REPO / "catalog"
SLEEP_S = 0.5  # politeness between API calls


def _get(url: str, cache_key: str) -> Any:
    """GET with on-disk provenance cache. Failures return None (recorded)."""
    CACHE.mkdir(exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", cache_key)[:180]
    cache_file = CACHE / f"{safe}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))["body"]
    time.sleep(SLEEP_S)
    req = urllib.request.Request(url, headers={"User-Agent": "llm-pairing-catalog/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print(f"    fetch failed: {url} ({exc})", file=sys.stderr)
        return None
    try:
        body = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        print(f"    non-JSON response: {url}", file=sys.stderr)
        return None
    cache_file.write_text(json.dumps({
        "url": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "body": body,
    }, ensure_ascii=False), encoding="utf-8")
    return body


def discover_repos(top: int) -> list[str]:
    q = urllib.parse.urlencode({
        "filter": "gguf", "sort": "downloads", "direction": "-1",
        "limit": str(top),
    })
    body = _get(f"{HF}/api/models?{q}", f"list_gguf_top{top}")
    if not isinstance(body, list):
        return []
    return [m["id"] for m in body if isinstance(m, dict) and m.get("id")]


def repo_detail(repo_id: str) -> dict[str, Any] | None:
    body = _get(f"{HF}/api/models/{repo_id}?blobs=true", f"model_{repo_id}")
    return body if isinstance(body, dict) else None


def base_config(base_id: str) -> dict[str, Any] | None:
    body = _get(f"{HF}/{base_id}/raw/main/config.json", f"config_{base_id}")
    return body if isinstance(body, dict) else None


def _base_model_of(detail: dict[str, Any]) -> str | None:
    card = detail.get("cardData") or {}
    base = card.get("base_model")
    if isinstance(base, list):
        base = base[0] if base else None
    if isinstance(base, str) and base.count("/") == 1:
        return base
    return None


def _ggufs(detail: dict[str, Any]) -> list[mapper.GgufFile]:
    files = []
    for s in detail.get("siblings") or []:
        name = s.get("rfilename", "")
        size = s.get("size")
        if name.lower().endswith(".gguf") and isinstance(size, int):
            files.append(mapper.GgufFile(filename=name, size=size))
    return files


def build(repos: list[str]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for repo_id in repos:
        print(f"-> {repo_id}", flush=True)
        detail = repo_detail(repo_id)
        if detail is None:
            skipped.append({"repo": repo_id, "reason": "repo detail fetch failed"})
            continue
        ggufs = _ggufs(detail)
        if not ggufs:
            skipped.append({"repo": repo_id, "reason": "no gguf files with sizes"})
            continue

        base_id = _base_model_of(detail)
        cfg_source = base_id or repo_id
        cfg = base_config(cfg_source)
        if cfg is None and base_id:
            cfg = base_config(repo_id)  # some repos ship their own config.json
            cfg_source = repo_id
        if cfg is None:
            skipped.append({"repo": repo_id,
                            "reason": f"config.json unavailable ({cfg_source}; "
                                      f"gated or absent)"})
            continue

        n_params: int | None = None
        if base_id:
            base_detail = repo_detail(base_id)
            if base_detail:
                st = base_detail.get("safetensors") or {}
                if isinstance(st.get("total"), int):
                    n_params = st["total"]

        model_id = base_id or repo_id
        result = mapper.map_config(model_id, cfg, ggufs=ggufs,
                                   n_params_total=n_params)
        if result.spec is None:
            skipped.append({"repo": repo_id, "reason": result.skip_reason or "?"})
            print(f"    skipped: {result.skip_reason}")
            continue
        card = detail.get("cardData") or {}
        entries.append({
            "spec": json.loads(result.spec.model_dump_json()),
            "meta": {
                "gguf_repo": repo_id,
                "config_source": cfg_source,
                "base_repo": base_id,
                "license": card.get("license"),
                "downloads": detail.get("downloads"),
                "mapping_notes": result.notes,
                "artifacts": result.artifacts,
            },
        })
        print(f"    ok: {result.spec.arch}/{result.spec.arch_handler}, "
              f"{len(result.spec.quants)} quants"
              + (f", notes={result.notes}" if result.notes else ""))

    # dedupe: the same base model often arrives via several GGUF repos
    # (observed: Qwen3.6-27B x3). Keep the repo with the most surviving
    # quants; record the losers as skipped duplicates (no silent drops).
    best: dict[str, dict[str, Any]] = {}
    for e in entries:
        mid = e["spec"]["model_id"]
        prev = best.get(mid)
        if prev is None or len(e["spec"]["quants"]) > len(prev["spec"]["quants"]):
            if prev is not None:
                skipped.append({"repo": prev["meta"]["gguf_repo"],
                                "reason": f"duplicate of {mid} (fewer quants)"})
            best[mid] = e
        else:
            skipped.append({"repo": e["meta"]["gguf_repo"],
                            "reason": f"duplicate of {mid} (fewer quants)"})
    entries = list(best.values())

    return {
        "catalog_schema": "1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "huggingface api (filter=gguf)",
        "entry_count": len(entries),
        "entries": entries,
        "skipped": skipped,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM pairing catalog builder")
    ap.add_argument("--top", type=int, default=25,
                    help="discover top-N GGUF repos by downloads (default 25)")
    ap.add_argument("--repos", help="comma-separated explicit repo list "
                                    "(skips discovery)")
    args = ap.parse_args()

    repos = ([r.strip() for r in args.repos.split(",")] if args.repos
             else discover_repos(args.top))
    if not repos:
        print("no repos to process (network problem?)", file=sys.stderr)
        return 1
    print(f"processing {len(repos)} repos ...")
    snapshot = build(repos)

    OUT_DIR.mkdir(exist_ok=True)
    # Reviewer item 3: date-only names overwrite same-day rebuilds, and
    # write_text's newline translation on Windows made file bytes diverge
    # from the recorded hash. Fix: content-addressed immutable name +
    # write_bytes (hash is of the EXACT bytes on disk).
    payload_bytes = json.dumps(snapshot, ensure_ascii=False, indent=1).encode("utf-8")
    digest = hashlib.sha256(payload_bytes).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_DIR / f"catalog-{stamp}-{digest[:8]}.json"
    out.write_bytes(payload_bytes)
    (OUT_DIR / f"catalog-{stamp}-{digest[:8]}.sha256").write_text(
        digest + "\n", encoding="utf-8", newline="\n"
    )
    print(f"\nwrote {out}  ({snapshot['entry_count']} entries, "
          f"{len(snapshot['skipped'])} skipped)")
    print(f"sha256 {digest}")
    print("skipped reasons:")
    for s in snapshot["skipped"]:
        print(f"  - {s['repo']}: {s['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
