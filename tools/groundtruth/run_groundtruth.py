"""Ground-truth collection harness (T-002 VERIFICATION L4 / S12).

Runs ON THE TARGET MACHINE. Talks only to the local Ollama server
(http://127.0.0.1:11434) and local system tools. Nothing is uploaded
anywhere; results land in data/groundtruth/<machine_id>/<run_id>.jsonl and
travel only when the owner commits them.

Per VERIFICATION §4.2, each cell records:
  1. pre-measurement VRAM snapshot (P-04: vram_free is a point-in-time value)
  2. actual load outcome: success / OOM / partial offload (via /api/ps
     size vs size_vram after generation)
  3. three timed generations with a fixed prompt (temperature 0) — raw
     numbers stored; median + IQR computed at report time (never mean)
  4. engine version, driver version, timestamps

Usage (PowerShell, from the repo root; Ollama must be running):
    python tools\\groundtruth\\run_groundtruth.py                 # all local models
    python tools\\groundtruth\\run_groundtruth.py --models gemma4:e4b --ctx 2048,8192
    python tools\\groundtruth\\run_groundtruth.py --runs 3 --num-predict 128

stdlib only — no third-party dependencies to install on the target machine.
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OLLAMA = "http://127.0.0.1:11434"
REPO = Path(__file__).resolve().parents[2]

# a deterministic ~500-token prompt so prefill numbers are comparable
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
          timeout: float = 600.0) -> dict[str, Any]:
    url = f"{OLLAMA}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out: dict[str, Any] = json.loads(resp.read().decode())
        return out


def _try(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def collect_machine_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.system().lower(),
        "platform_release": platform.release(),
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    # RAM (Windows: wmic may be absent on 11+; try PowerShell CIM)
    ps = _try(["powershell", "-NoProfile", "-Command",
               "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
    if ps and ps.isdigit():
        info["ram_total_bytes"] = int(ps)
    gpu = _try(["nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits"])
    if gpu:
        name, mtotal, mfree, drv = [s.strip() for s in gpu.split(",")]
        info["gpu"] = {
            "name": name,
            "vram_total_mib": int(mtotal),
            "vram_free_mib": int(mfree),
            "driver": drv,
            "source": "nvidia-smi",
        }
    else:
        info["gpu"] = None  # iGPU/Apple/none — recorded honestly as unknown here
    try:
        info["ollama_version"] = _http("/api/version", timeout=5).get("version")
    except Exception:
        info["ollama_version"] = None
    return info


def vram_snapshot() -> dict[str, Any] | None:
    out = _try(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits"])
    if not out:
        return None
    used, free = [int(s.strip()) for s in out.split(",")]
    return {"vram_used_mib": used, "vram_free_mib": free,
            "at": datetime.now(timezone.utc).isoformat()}


def ollama_ps() -> list[dict[str, Any]]:
    try:
        models = _http("/api/ps", timeout=10).get("models", [])
        return [
            {"name": m.get("name"), "size": m.get("size"),
             "size_vram": m.get("size_vram")}
            for m in models
        ]
    except Exception:
        return []


def run_cell(model: str, num_ctx: int, runs: int, num_predict: int) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "model": model,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "vram_before": vram_snapshot(),
        "runs": [],
    }
    try:
        show = _http("/api/show", {"model": model}, timeout=30)
        cell["model_details"] = show.get("details")
        cell["model_info"] = {
            k: v for k, v in (show.get("model_info") or {}).items()
            if isinstance(v, (int, float, str, bool))
        }
    except Exception as e:
        cell["model_details_error"] = str(e)[:300]

    for i in range(runs + 1):  # +1: first generation is the load/warmup run
        t0 = time.monotonic()
        try:
            # vary the prompt head per run: an identical prompt hits Ollama's
            # prompt cache and produces absurd prefill numbers (observed
            # 13k tok/s on a CPU box). Distinct tokens force a real prefill.
            r = _http("/api/generate", {
                "model": model,
                "prompt": f"Session {i}: {PROMPT}",
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                    "temperature": 0,
                    "seed": 42,
                },
            })
            rec: dict[str, Any] = {
                "kind": "warmup" if i == 0 else "measure",
                "wall_s": round(time.monotonic() - t0, 3),
                "load_duration_ns": r.get("load_duration"),
                "prompt_eval_count": r.get("prompt_eval_count"),
                "prompt_eval_duration_ns": r.get("prompt_eval_duration"),
                "eval_count": r.get("eval_count"),
                "eval_duration_ns": r.get("eval_duration"),
            }
            if rec["eval_count"] and rec["eval_duration_ns"]:
                rec["decode_tps"] = round(
                    rec["eval_count"] / rec["eval_duration_ns"] * 1e9, 3
                )
            if rec["prompt_eval_count"] and rec["prompt_eval_duration_ns"]:
                rec["prefill_tps"] = round(
                    rec["prompt_eval_count"] / rec["prompt_eval_duration_ns"] * 1e9, 3
                )
            cell["runs"].append(rec)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:500]
            except Exception:
                pass
            cell["runs"].append({
                "kind": "warmup" if i == 0 else "measure",
                "error": f"HTTP {e.code}", "error_body": body,
            })
            cell["outcome"] = "ERROR_OR_OOM"
            break
        except Exception as e:
            cell["runs"].append({"kind": "error", "error": str(e)[:300]})
            cell["outcome"] = "ERROR_OR_OOM"
            break

    cell["ps_after"] = ollama_ps()  # size vs size_vram exposes partial offload
    for p in cell["ps_after"]:
        if p.get("name", "").startswith(model.split(":")[0]) and p.get("size"):
            if p.get("size_vram") is not None:
                frac = p["size_vram"] / p["size"] if p["size"] else None
                cell["vram_fraction"] = round(frac, 4) if frac is not None else None
    if "outcome" not in cell:
        measures = [r for r in cell["runs"] if r.get("kind") == "measure"
                    and "decode_tps" in r]
        if measures:
            frac = cell.get("vram_fraction")
            cell["outcome"] = (
                "SUCCESS_FULL_GPU" if frac is not None and frac >= 0.999
                else "SUCCESS_PARTIAL_OFFLOAD" if frac is not None and frac > 0
                else "SUCCESS_CPU_ONLY" if frac == 0.0
                else "SUCCESS_PLACEMENT_UNKNOWN"
            )
        else:
            cell["outcome"] = "NO_MEASUREMENTS"
    cell["vram_after"] = vram_snapshot()
    return cell


def local_models() -> list[str]:
    tags = _http("/api/tags", timeout=10).get("models", [])
    # cloud-proxied models are not local ground truth — skip them
    return [m["name"] for m in tags
            if not str(m.get("name", "")).endswith(":cloud")]


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM pairing ground-truth harness")
    ap.add_argument("--models", help="comma-separated Ollama model names "
                                     "(default: every local model)")
    ap.add_argument("--ctx", default="2048,8192",
                    help="comma-separated num_ctx values (default 2048,8192)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--num-predict", type=int, default=128)
    args = ap.parse_args()

    try:
        _http("/api/version", timeout=5)
    except Exception:
        print("ERROR: Ollama server not reachable on 127.0.0.1:11434 — "
              "start the Ollama app first.", file=sys.stderr)
        return 1

    models = ([m.strip() for m in args.models.split(",")]
              if args.models else local_models())
    if not models:
        print("No local models found (cloud models are skipped).", file=sys.stderr)
        return 1
    ctxs = [int(c) for c in args.ctx.split(",")]

    machine = collect_machine_info()
    machine_id = machine["hostname"].lower().replace(" ", "-")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "data" / "groundtruth" / machine_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.jsonl"

    print(f"machine: {machine_id}  ollama {machine.get('ollama_version')}")
    print(f"models: {models}  ctx: {ctxs}  runs: {args.runs}+warmup")
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "machine", **machine},
                           ensure_ascii=False) + "\n")
        for model in models:
            for ctx in ctxs:
                print(f"-> {model} @ ctx {ctx} ...", flush=True)
                cell = run_cell(model, ctx, args.runs, args.num_predict)
                f.write(json.dumps({"kind": "cell", **cell},
                                   ensure_ascii=False) + "\n")
                f.flush()
                tps = [r.get("decode_tps") for r in cell["runs"]
                       if r.get("kind") == "measure" and r.get("decode_tps")]
                print(f"   {cell['outcome']}  decode_tps={tps}  "
                      f"vram_fraction={cell.get('vram_fraction')}")
    print(f"\nwrote {out_path}")
    print("Next: hand this file back so predictions can be scored against it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
