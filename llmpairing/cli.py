"""Unified CLI (review #2/#3 naming + one-command flow).

    llmpairing probe [--json PATH]      -> the T-001 read-only scan
    llmpairing recommend [options]      -> scan (or load) + match + Top-3

`recommend` is the end-to-end path: probe the machine (or take
--profile), load the newest sidecar-verified catalog snapshot, run the
T-002/T-003 pure pipeline per scenario, rank with llmpairing.recommend,
and print actionable cards — plain-language verdicts, predicted speed
with its evidence tier, and copy-ready `ollama` / `llama.cpp` commands.

I/O lives here; every decision is made by the tested pure core.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from llmpairing.budget.classify import classify_fit
from llmpairing.predict.calibration import MachineCalibration
from llmpairing.predict.decode import decode_bytes_per_token, predict_decode
from llmpairing.recommend import RecCandidate, RecResult, recommend
from llmpairing.schemas import HardwareProfile, ModelSpec, Workload

PREVIEW_QUANTS = ("Q4_K_M", "Q5_K_M", "Q8_0", "F16")

#: plain-language verdict labels (kept in sync with the demo template)
VERDICT_LABEL = {
    "FITS": "可以跑",
    "TIGHT": "勉強可跑",
    "RAM_ONLY": "可跑（無顯卡）",
    "PARTIAL_OFFLOAD": "可跑・會變慢",
    "OOM_AT_CONTEXT": "長文跑不動",
    "OOM_AT_LOAD": "跑不動",
}
KIND_LABEL = {"CAPABILITY": "🏆 能力優先", "SPEED": "⚡ 速度優先",
              "LONG_CONTEXT": "📜 長文優先"}


def gather_candidates(hw: HardwareProfile,
                      entries: list[tuple[ModelSpec, dict[str, Any]]],
                      *, calibration: MachineCalibration | None,
                      target_ctx: int, long_ctx: int) -> list[RecCandidate]:
    """Run the pure pipeline over catalog entries for one machine scenario."""
    cands: list[RecCandidate] = []
    for spec, _meta in entries:
        shown = ([q for q in spec.quants if q.label in PREVIEW_QUANTS]
                 or spec.quants[:2])
        for q in shown:
            for ctx in (target_ctx, long_ctx):
                wl = Workload(ctx_target_tokens=ctx, kv_dtype="f16")
                fit = classify_fit(hw, spec, q, wl)
                tps: float | None = None
                tps_tier: str | None = None
                if fit.budget_bytes.value is not None:
                    try:
                        t = predict_decode(hw, spec, q, wl, fit,
                                           calibration=calibration)
                        tv = t.decode_tps_at_target_ctx
                        if tv.value is not None:
                            tps = round(tv.value, 1)
                            tps_tier = tv.tier.value
                    except ValueError:
                        pass  # V-P5 sentinel: no speed claim for this cell
                try:
                    bpt: float | None = decode_bytes_per_token(spec, q, wl, ctx)
                except Exception:
                    bpt = None  # unrankable, never guessed (R-2)
                cands.append(RecCandidate(
                    model_id=spec.model_id,
                    label=spec.model_id.split("/")[-1],
                    quant=q.label,
                    params_active=spec.n_params_active,
                    params_total=spec.n_params_total,
                    weights_bytes=q.file_bytes,
                    verdict=fit.verdict.value,
                    ctx=ctx,
                    ctx_max=fit.ctx_max_tokens.value,
                    bytes_per_token=bpt,
                    tps=tps,
                    tps_tier=tps_tier,
                ))
    return cands


def format_recommendation(machine_label: str, rr: RecResult,
                          repo_of: dict[str, str], *, catalog_name: str,
                          calibrated: bool) -> str:
    """Render one scenario's picks as terminal text. Pure."""
    lines = [f"◆ {machine_label}"]
    if not rr.picks:
        for n in rr.notes:
            lines.append(f"  {n}")
        if not rr.notes:
            lines.append("  目錄中沒有可完整載入的模型——不勉強推薦（R-2）")
    for p in rr.picks:
        c = p.candidate
        v = VERDICT_LABEL.get(c.verdict, c.verdict)
        gib = (f"{c.weights_bytes / 1024**3:.2f} GiB"
               if c.weights_bytes else "?")
        tps = (f"・預估 ~{c.tps:.0f} tok/s（{c.tps_tier or 'T0'}）"
               if c.tps is not None else "")
        lines.append("")
        lines.append(f"  {KIND_LABEL.get(p.kind, p.kind)}  {c.label}")
        lines.append(f"    {c.quant}・{gib}・active "
                     f"{c.params_active / 1e9:.1f}B・context {c.ctx:,}"
                     f"・{v}{tps}")
        lines.append(f"    {p.reason}")
        for cav in p.caveats:
            lines.append(f"    ⚠ {cav}")
        repo = repo_of.get(c.model_id)
        if repo:
            lines.append(f"    $ ollama run hf.co/{repo}:{c.quant}")
            lines.append(f"    $ llama-server -hf {repo}:{c.quant}")
    for n in rr.notes:
        if rr.picks:
            lines.append(f"  · {n}")
    lines.append("")
    lines.append(f"  目錄：{catalog_name}・記憶體判定 T0（規格估算）・速度 "
                 + ("T1（本機已校準）" if calibrated
                    else "未經本機校準（可跑 T-003 校準升級）"))
    return "\n".join(lines)


def _load_catalog(catalog_dir: Path) -> tuple[str, list[tuple[ModelSpec, dict[str, Any]]]]:
    snaps = sorted(catalog_dir.glob("catalog-*.json"))
    if not snaps:
        raise SystemExit(f"找不到目錄快照（{catalog_dir}/catalog-*.json）——"
                         f"先執行 tools/catalog/build_catalog.py")
    raw = snaps[-1].read_bytes()
    sidecar = snaps[-1].with_suffix(".sha256")
    if sidecar.exists():
        want = sidecar.read_text(encoding="utf-8").strip()
        got = hashlib.sha256(raw).hexdigest()
        if want != got:
            raise SystemExit(
                f"catalog integrity FAILURE: {snaps[-1].name} sha256 "
                f"{got[:12]}… != sidecar {want[:12]}… — 拒絕使用毀損資料")
    data = json.loads(raw.decode("utf-8"))
    return snaps[-1].name, [
        (ModelSpec.model_validate(e["spec"]), e.get("meta", {}))
        for e in data.get("entries", [])
    ]


def _load_calibration(cal_dir: Path) -> MachineCalibration | None:
    cals = sorted(cal_dir.glob("*.json"))
    if not cals:
        return None
    return MachineCalibration.model_validate_json(
        cals[-1].read_text(encoding="utf-8"))


def _recommend_cmd(args: argparse.Namespace) -> int:
    if args.profile:
        prof_path = Path(args.profile)
        if not prof_path.exists():
            raise SystemExit(f"--profile {args.profile} 不存在")
        hw = HardwareProfile.model_validate(
            json.loads(prof_path.read_text(encoding="utf-8")))
    else:
        from llmpairing.probe import assemble
        from llmpairing.probe.cli import _current_platform
        from llmpairing.probe.runner import ProbeError
        diags: list[str] = []
        try:
            hw = assemble.build_profile(
                diags, platform_name=_current_platform(),
                measured_at_unix=int(time.time()))
        except ProbeError as exc:
            print(f"掃描失敗: {exc}", file=sys.stderr)
            return 1

    catalog_name, entries = _load_catalog(Path(args.catalog_dir))
    calibration = _load_calibration(Path(args.calibration_dir))
    repo_of = {spec.model_id: str(meta.get("gguf_repo") or "")
               for spec, meta in entries}

    scenarios: list[tuple[str, HardwareProfile, bool]] = []
    if any(a.topology == "INTEGRATED_SHARED" for a in hw.accelerators):
        scenarios.append(("你的機器（若引擎用 iGPU）", hw, False))
        scenarios.append(("你的機器（純 CPU，如 Ollama）",
                          hw.model_copy(update={"accelerators": []}), True))
    else:
        scenarios.append(("你的機器", hw, not hw.accelerators))

    print("LLM pairing — 為這台機器推薦（誠實優先：不知道就說不知道）\n")
    for label, scen_hw, is_cpu_pool in scenarios:
        cal = calibration if (calibration is not None
                              and calibration.pool == ("cpu" if is_cpu_pool
                                                       else "gpu")) else None
        cands = gather_candidates(scen_hw, entries, calibration=cal,
                                  target_ctx=args.ctx, long_ctx=args.long_ctx)
        rr = recommend(cands, target_ctx=args.ctx, long_ctx=args.long_ctx)
        print(format_recommendation(label, rr, repo_of,
                                    catalog_name=catalog_name,
                                    calibrated=cal is not None))
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="llmpairing",
        description="掃描這台電腦，誠實配對可在本機執行的 LLM")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("probe", add_help=False,
                   help="唯讀硬體掃描（等同 llmpairing-probe）")

    rec = sub.add_parser("recommend", help="掃描 + 配對 + Top-3 推薦")
    rec.add_argument("--profile", help="改用既有的 HardwareProfile JSON "
                                       "（省略則現場掃描）")
    rec.add_argument("--catalog-dir", default="catalog",
                     help="目錄快照資料夾（預設 ./catalog）")
    rec.add_argument("--calibration-dir", default="calibration",
                     help="T-003 校準資料夾（預設 ./calibration）")
    rec.add_argument("--ctx", type=int, default=8_192,
                     help="目標 context（預設 8192）")
    rec.add_argument("--long-ctx", type=int, default=32_768,
                     help="長文 context（預設 32768）")

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "probe":
        from llmpairing.probe.cli import main as probe_main
        code = probe_main(argv[1:])
        raise SystemExit(code)
    args = ap.parse_args(argv)
    if args.command == "recommend":
        return _recommend_cmd(args)
    return 0  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
