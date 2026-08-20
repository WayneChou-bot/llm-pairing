"""Machine calibration runner (T-003 MVP). RUNS ON THE OWNER'S MACHINE.

Loads a WHITELISTED model in Ollama, times N generations, and computes the
measured BW x eta product:  product_i = decode_tps_i x bytes_per_token(ctx_i).
Writes calibration/<machine_id>.json (median + min/max over runs, full
provenance). Nothing is uploaded anywhere.

The calibration model must exist in the newest catalog snapshot (its
bytes-per-token model must be trusted — honesty rule 1); unsupported
architectures like gemma4 are refused.

Usage (repo root, conda env active, Ollama running):
    python tools\\calibrate\\run_calibration.py --ollama-model qwen3.5:4b \\
        --catalog-id Qwen/Qwen3.5-4B --quant Q4_K_M
"""
from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from llmpairing.predict.calibration import MachineCalibration  # noqa: E402
from llmpairing.predict.decode import _bytes_per_token  # noqa: E402
from llmpairing.schemas import ModelSpec, Workload  # noqa: E402

OLLAMA = "http://127.0.0.1:11434"

PROMPT = (
    "You are a careful assistant. Summarize the following notes in detail.\n"
    + "\n".join(
        f"Note {i}: the quick brown fox jumps over the lazy dog near "
        f"station {i}, carrying package number {i * 7} to the depot."
        for i in range(1, 61)
    )
    + "\nWrite a thorough summary:"
)


def _http(path: str, payload: dict[str, Any] | None = None,
          timeout: float = 900.0) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{OLLAMA}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out: dict[str, Any] = json.loads(resp.read().decode())
        return out


def load_catalog_model(catalog_id: str, quant_label: str) -> tuple[ModelSpec, Any]:
    snaps = sorted((REPO / "catalog").glob("catalog-*.json"))
    if not snaps:
        raise SystemExit("no catalog snapshot — run tools/catalog/build_catalog.py first")
    data = json.loads(snaps[-1].read_text(encoding="utf-8"))
    for e in data["entries"]:
        if e["spec"]["model_id"] == catalog_id:
            spec = ModelSpec.model_validate(e["spec"])
            if not spec.arch_supported:
                raise SystemExit(f"{catalog_id}: arch not whitelisted — cannot "
                                 f"be a calibration model (honesty rule 1)")
            for q in spec.quants:
                if q.label == quant_label:
                    return spec, q
            raise SystemExit(f"{catalog_id}: quant {quant_label} not in snapshot")
    raise SystemExit(f"{catalog_id} not found in {snaps[-1].name}")


def detect_pool(ps_models: list[dict[str, Any]], ollama_model: str) -> str:
    """Review #5 P2: label the memory pool from THIS run's model only.

    The pool label decides which scenarios this calibration may ever
    apply to (strict pool gate), so it must come from the calibration
    model's own placement — background models loaded in Ollama used to
    vote via the old loop and could win. Partial offload is refused
    outright: a product measured across two memory pools would poison
    both scenarios (R-2: refuse, never the old ">50% VRAM" guess).
    """
    mine = [p for p in ps_models
            if p.get("name") == ollama_model or p.get("model") == ollama_model]
    if not mine:
        raise ValueError(
            f"calibration model {ollama_model!r} not found in ollama "
            "/api/ps — cannot verify placement, refusing to label the pool")
    entry = mine[0]
    size = int(entry.get("size") or 0)
    vram = int(entry.get("size_vram") or 0)
    if size <= 0:
        raise ValueError(f"ollama /api/ps reports no size for "
                         f"{ollama_model!r} — cannot verify placement")
    if vram == 0:
        return "cpu"
    if vram >= size:
        return "gpu"
    raise ValueError(
        f"{ollama_model!r} is partially offloaded ({vram}/{size} bytes in "
        "VRAM) — a calibration product measured across two memory pools "
        "would poison both scenarios; re-run fully on CPU (e.g. "
        "num_gpu=0) or fully on GPU")


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM pairing calibration runner")
    ap.add_argument("--ollama-model", required=True)
    ap.add_argument("--catalog-id", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--num-predict", type=int, default=128)
    ap.add_argument("--num-ctx", type=int, default=4096)
    args = ap.parse_args()

    spec, quant = load_catalog_model(args.catalog_id, args.quant)
    wl = Workload(ctx_target_tokens=args.num_ctx)

    try:
        _http("/api/version", timeout=5)
    except Exception:
        print("ERROR: Ollama not reachable on 127.0.0.1:11434", file=sys.stderr)
        return 1

    print(f"calibrating on {args.ollama_model} "
          f"(mapped to {spec.model_id} {args.quant}, arch {spec.arch}) ...")
    products: list[float] = []
    tps_seen: list[float] = []
    ctx_mids: list[int] = []
    pool: str | None = None
    for i in range(args.runs + 1):  # first run = load/warmup, not scored
        r = _http("/api/generate", {
            "model": args.ollama_model,
            "prompt": f"Session {i}: {PROMPT}",
            "stream": False, "keep_alive": "10m",
            "options": {"num_ctx": args.num_ctx,
                        "num_predict": args.num_predict,
                        "temperature": 0, "seed": 42},
        })
        if i == 0:
            # review #5 P2: pool detection consults THIS model's own
            # placement, runs even if the warmup lacks timings, and
            # refuses ambiguity instead of guessing
            ps = _http("/api/ps", timeout=10).get("models", [])
            try:
                pool = detect_pool(ps, args.ollama_model)
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
        ec, ed = r.get("eval_count"), r.get("eval_duration")
        pc = r.get("prompt_eval_count") or 0
        if not ec or not ed:
            print(f"  run {i}: no timings, skipped")
            continue
        tps = ec / ed * 1e9
        # ctx during decode grows from pc to pc+ec: use the midpoint (rule 5)
        ctx_mid = int(pc + ec / 2)
        bpt = _bytes_per_token(spec, quant, wl, ctx_mid)
        if i == 0:
            print(f"  warmup: {tps:.2f} tok/s (not scored, pool={pool})")
            continue
        products.append(tps * bpt)
        tps_seen.append(round(tps, 2))
        ctx_mids.append(ctx_mid)
        print(f"  run {i}: {tps:.2f} tok/s x bpt({ctx_mid}) "
              f"-> product {tps * bpt / 1e9:.1f} GB/s")

    if len(products) < 2:
        print("not enough scored runs", file=sys.stderr)
        return 1

    # review #4 P1: machine_id is the same one-way fingerprint the probe
    # stamps on HardwareProfile — calibration only applies when they match
    from llmpairing.probe.cpu import collect_cpu
    from llmpairing.probe.fingerprint import machine_fingerprint
    from llmpairing.probe.memory import collect_memory

    if pool is None:  # detection is mandatory — no pool label, no file
        print("ERROR: pool never determined (warmup did not run?) — "
              "refusing to write a calibration", file=sys.stderr)
        return 1

    _diags: list[str] = []
    machine_id = machine_fingerprint(
        socket.gethostname(), collect_cpu(_diags).model,
        collect_memory(_diags).total_bytes)
    cal = MachineCalibration(
        schema_version="1",
        machine_id=machine_id,
        pool=pool,  # type: ignore[arg-type]
        backend="ollama",
        product_bytes_per_s=statistics.median(products),
        product_ci_low=min(products),
        product_ci_high=max(products),
        n_runs=len(products),
        calibration_model=spec.model_id,
        calibration_quant=args.quant,
        calibration_ctx=int(statistics.median(ctx_mids)),
        measured_at_unix=int(time.time()),
        notes=["CALIBRATION_CTX_APPROX",
               f"raw_tps={tps_seen}",
               f"ollama_model={args.ollama_model}",
               f"recorded_at={datetime.now(timezone.utc).isoformat()}"],
    )
    out_dir = REPO / "calibration"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{machine_id}.json"
    out.write_text(cal.model_dump_json(indent=2), encoding="utf-8")
    print(f"\npool={pool}  product median {cal.product_bytes_per_s/1e9:.1f} GB/s "
          f"[{cal.product_ci_low/1e9:.1f} – {cal.product_ci_high/1e9:.1f}]")
    print(f"wrote {out}")
    print("預測現已可升級為 T1：重跑 python tools\\demo\\build_demo.py 即生效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
