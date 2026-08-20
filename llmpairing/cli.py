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
from llmpairing.catalog.mapper import classify_source
from llmpairing.predict.calibration import MachineCalibration, pick_calibration
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
    "CTX_EXCEEDS_MODEL_MAX": "超過模型 context 上限",
}
KIND_LABEL = {"CAPABILITY": "🏆 能力優先", "SPEED": "⚡ 速度優先",
              "LONG_CONTEXT": "📜 長文優先"}
TRUST_LABEL = {"official": "官方", "trusted_quantizer": "可信量化者",
               "community": "社群"}
CAVEAT_TEXT = {
    "TIGHT_FALLBACK": "TIGHT：剩餘記憶體低於安全緩衝——其他程式一搶記憶體就可能失敗",
    "SPEED_RELATIVE_ORDER": "速度為相對排序（依每 token 記憶體流量）；跑一次 T-003 校準可升級為絕對值",
    "COMMUNITY_SOURCE": "社群來源（非官方/可信量化者）——內容與品質未經審核",
}


def note_text(code: str, params: dict[str, object]) -> str:
    ctx = params.get("ctx", "?")
    if code == "NO_FIT_AT_TARGET":
        return (f"在 context {ctx} 下，目錄中沒有能完整載入這台機器的模型"
                "——不勉強推薦（R-2）")
    if code == "NO_FIT_AT_LONG":
        return f"context {ctx} 下沒有可完整載入的組合——長文需求請降低 context"
    if code == "COMMUNITY_EXCLUDED":
        return (f"已排除 {params.get('count', '?')} 個社群來源模型"
                "（--include-community 可納入）")
    if code == "CTX_EXCEEDS_MODEL_MAX":
        return (f"其中 {params.get('count', '?')} 個模型的 context 上限低於"
                f"你要求的 {ctx}——這不是記憶體問題，任何硬體都跑不了；"
                "請把 --ctx 降到模型上限以內")
    return f"{code} {params}"


def source_of(meta: dict[str, Any]) -> dict[str, Any]:
    """Trust info from meta, backfilled for pre-review-#4 snapshots."""
    if meta.get("trust"):
        return {"trust": meta["trust"], "publisher": meta.get("publisher"),
                "relation": meta.get("relation"),
                "content_tags": meta.get("content_tags") or []}
    repo = meta.get("gguf_repo")
    if repo:
        return classify_source(str(repo), meta.get("base_repo"), None, [])
    return {"trust": "community", "publisher": None, "relation": None,
            "content_tags": []}


def gather_candidates(hw: HardwareProfile,
                      entries: list[tuple[ModelSpec, dict[str, Any]]],
                      *, calibration: MachineCalibration | None,
                      target_ctx: int, long_ctx: int) -> list[RecCandidate]:
    """Run the pure pipeline over catalog entries for one machine scenario."""
    cands: list[RecCandidate] = []
    for spec, _meta in entries:
        src = source_of(_meta)
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
                    trust=str(src["trust"]),
                    flags=fit.flags,
                ))
    return cands


def _scenario_calibration(
    scen_hw: HardwareProfile, calibrations: list[MachineCalibration],
    pool_name: str,
) -> tuple[MachineCalibration | None, str | None]:
    """Pick + gate a calibration for one scenario (review #5 P1).

    Returns (calibration-to-apply | None, footer_flag | None). The
    identity gate is the same calibration_applies used inside predict —
    the CLI never hands predict a file the gate would refuse, so the
    footer and the per-cell math can no longer disagree. A pool mismatch
    is the normal dual-scenario state (quietly uncalibrated); machine-
    identity refusals are surfaced so the footer can say WHY there is no
    T1 — a downloaded repo carrying someone else's calibration file must
    never render 「本機已校準」.
    """
    from llmpairing.predict.calibration import calibration_applies
    chosen = pick_calibration(scen_hw, calibrations)
    if chosen is None:
        return None, None
    applies, flag = calibration_applies(scen_hw, chosen, pool_name)
    if applies:
        return chosen, None
    if flag == "CALIBRATION_POOL_MISMATCH_IGNORED":
        return None, None
    return None, flag


def format_recommendation(machine_label: str, rr: RecResult,
                          repo_of: dict[str, str], *, catalog_name: str,
                          calibrated: bool,
                          cal_ignored_flag: str | None = None) -> str:
    """Render one scenario's picks as terminal text. Pure."""
    lines = [f"◆ {machine_label}"]
    if not rr.picks:
        for n in rr.notes:
            lines.append(f"  {note_text(n.code, dict(n.params))}")
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
        src_bits = [TRUST_LABEL.get(c.trust, c.trust)]
        lines.append(f"    來源信任：{'・'.join(src_bits)}")
        for cav in p.caveats:
            lines.append(f"    ⚠ {CAVEAT_TEXT.get(cav, cav)}")
        for fl in p.candidate.flags:
            if fl in ("NVIDIA_SMI_PARSER_UNVERIFIED_ON_REAL_HW",
                      "DISCRETE_GPU_DATA_UNAVAILABLE_EXCLUDED",
                      "UBATCH_NOT_YET_MODELED"):
                lines.append(f"    ⚠ 旗標：{fl}")
        repo = repo_of.get(c.model_id)
        if repo:
            lines.append(f"    $ ollama run hf.co/{repo}:{c.quant}")
            lines.append(f"    $ llama-server -hf {repo}:{c.quant}")
    for n in rr.notes:
        if rr.picks:
            lines.append(f"  · {note_text(n.code, dict(n.params))}")
    lines.append("")
    if calibrated:
        speed_txt = "T1（本機已校準）"
    elif cal_ignored_flag == "CALIBRATION_MACHINE_MISMATCH_IGNORED":
        speed_txt = ("T0——校準檔與本機指紋不符，已忽略"
                     "（在這台機器重跑 T-003 校準即可升級）")
    elif cal_ignored_flag == "CALIBRATION_PROFILE_HAS_NO_MACHINE_ID":
        speed_txt = ("T0——硬體檔缺機器指紋，校準檔已忽略"
                     "（重跑 llmpairing probe 產生新檔即可）")
    else:
        speed_txt = "未經本機校準（可跑 T-003 校準升級）"
    lines.append(f"  目錄：{catalog_name}・記憶體判定 T0（規格估算）・速度 "
                 + speed_txt)
    return "\n".join(lines)


def _load_catalog(catalog_dir: Path, *, allow_fallback: bool = False,
                  ) -> tuple[str, list[tuple[ModelSpec, dict[str, Any]]]]:
    snaps = sorted(catalog_dir.glob("catalog-*.json"))
    # review #5 P2: first-run UX — with the DEFAULT ./catalog, an
    # editable install run from any cwd falls back to the repo
    # checkout's own bundled snapshots instead of failing about the
    # working directory. An explicit --catalog-dir is always respected
    # (a typo must fail loudly, never silently use other data).
    fallback = Path(__file__).resolve().parents[1] / "catalog"
    searched = f"{catalog_dir.resolve()}/catalog-*.json"
    if not snaps and allow_fallback and fallback != catalog_dir.resolve():
        snaps = sorted(fallback.glob("catalog-*.json"))
        searched += f"，也查過 repo 內建的 {fallback}"
    if not snaps:
        raise SystemExit(
            f"找不到目錄快照（{searched}）。\n"
            f"請用 --catalog-dir 指定快照資料夾；"
            f"還沒有快照就先跑 tools/catalog/build_catalog.py")
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


def _load_calibrations(cal_dir: Path) -> list[MachineCalibration]:
    return [
        MachineCalibration.model_validate_json(f.read_text(encoding="utf-8"))
        for f in sorted(cal_dir.glob("*.json"))
    ]


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

    catalog_name, entries = _load_catalog(
        Path(args.catalog_dir),
        allow_fallback=args.catalog_dir == "catalog")
    calibrations = _load_calibrations(Path(args.calibration_dir))
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
        # review #5 P1: gate BEFORE handing to predict; the footer claim
        # is then derived from what the picks actually carry (evidence),
        # never from mere file presence
        cal, cal_flag = _scenario_calibration(
            scen_hw, calibrations, "cpu" if is_cpu_pool else "gpu")
        cands = gather_candidates(scen_hw, entries, calibration=cal,
                                  target_ctx=args.ctx, long_ctx=args.long_ctx)
        rr = recommend(cands, target_ctx=args.ctx, long_ctx=args.long_ctx,
                       include_community=args.include_community)
        calibrated = any(p.candidate.tps_tier == "T1" for p in rr.picks)
        print(format_recommendation(label, rr, repo_of,
                                    catalog_name=catalog_name,
                                    calibrated=calibrated,
                                    cal_ignored_flag=cal_flag))
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
    rec.add_argument("--include-community", action="store_true",
                     help="把社群來源（非官方/可信量化者）納入推薦池；"
                          "預設排除並顯示排除數")

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
