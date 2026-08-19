"""Recommendation policy — a PURE ranking over precomputed cells (R-1).

The demo (and any future front end) computes fit + speed per combo with
the T-002/T-003 pipeline, then hands plain candidate records here; this
module only orders them and writes down why. It never probes, classifies
or predicts.

Honesty contract (tests pin all of it):
- recommendable = FITS / RAM_ONLY. TIGHT is a flagged fallback used only
  when nothing fits comfortably. PARTIAL_OFFLOAD and worse are never
  recommended: "runs, but N x slower" is a cliff, not a recommendation.
- capability ranking = active params (the only capability proxy we can
  cite without computing quality ourselves; quality is cite-not-compute).
- speed ranking uses predicted tps when present; otherwise it falls back
  to bytes-per-token, which orders correctly under the roofline model
  even when absolute tok/s is unknown — and the caveat says exactly that.
  Candidates with neither are skipped for speed, never guessed (R-2).
- an empty pool returns empty picks plus an explicit note.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

_RUNNABLE = ("FITS", "RAM_ONLY")


class RecCandidate(BaseModel):
    """One (model, quant) evaluated at one ctx on one machine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    label: str
    quant: str
    params_active: int
    params_total: int
    weights_bytes: int | None
    verdict: str
    ctx: int
    ctx_max: int | None
    bytes_per_token: float | None
    tps: float | None
    tps_tier: str | None


class Pick(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["CAPABILITY", "SPEED", "LONG_CONTEXT"]
    candidate: RecCandidate
    reason: str
    caveats: list[str] = []


class RecResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    picks: list[Pick] = []
    notes: list[str] = []


def _pool(cands: list[RecCandidate], ctx: int) -> tuple[list[RecCandidate], list[str]]:
    """Runnable candidates at ctx; TIGHT fallback carries a caveat."""
    run = [c for c in cands if c.ctx == ctx and c.verdict in _RUNNABLE]
    if run:
        return run, []
    tight = [c for c in cands if c.ctx == ctx and c.verdict == "TIGHT"]
    if tight:
        return tight, [
            "TIGHT：剩餘記憶體低於安全緩衝——其他程式一搶記憶體就可能失敗"
        ]
    return [], []


def _by_capability(c: RecCandidate) -> tuple[int, int]:
    return (c.params_active, c.weights_bytes or 0)


def recommend(cands: list[RecCandidate], *, target_ctx: int = 8_192,
              long_ctx: int = 32_768) -> RecResult:
    picks: list[Pick] = []
    notes: list[str] = []

    pool, pool_caveats = _pool(cands, target_ctx)
    if not pool:
        notes.append(
            f"在 context {target_ctx:,} 下，目錄中沒有能完整載入這台機器的"
            "模型——不勉強推薦（R-2）"
        )
        return RecResult(picks=[], notes=notes)

    # -- capability: largest active params, tie -> larger weights (higher quant)
    cap = max(pool, key=_by_capability)
    picks.append(Pick(
        kind="CAPABILITY",
        candidate=cap,
        reason=(f"可完整載入的最大模型（active "
                f"{cap.params_active / 1e9:.1f}B 參數）"),
        caveats=list(pool_caveats),
    ))

    # -- speed: tps when predicted, else bytes/token physics (relative order)
    with_tps = [c for c in pool if c.tps is not None]
    speed: RecCandidate | None = None
    speed_caveats = list(pool_caveats)
    if with_tps:
        ranked = sorted(with_tps, key=lambda c: c.tps or 0.0, reverse=True)
    else:
        rankable = [c for c in pool if c.bytes_per_token is not None]
        ranked = sorted(rankable, key=lambda c: c.bytes_per_token or 0.0)
        if ranked:
            speed_caveats.append(
                "速度為相對排序（依每 token 記憶體流量的物理量）；"
                "絕對 tok/s 未校準——跑一次 T-003 校準即可升級"
            )
    if ranked:
        distinct = [c for c in ranked if c.model_id != cap.model_id]
        speed = distinct[0] if distinct else ranked[0]
        tps_txt = (f"預估 ~{speed.tps:.0f} tok/s" if speed.tps is not None
                   else "每 token 記憶體流量最低")
        picks.append(Pick(
            kind="SPEED", candidate=speed,
            reason=f"可完整載入組合中最快（{tps_txt}）",
            caveats=speed_caveats,
        ))

    # -- long context: capability ranking at the long ctx point
    long_pool, long_caveats = _pool(cands, long_ctx)
    if long_pool:
        lc = max(long_pool, key=_by_capability)
        picks.append(Pick(
            kind="LONG_CONTEXT", candidate=lc,
            reason=(f"在 context {long_ctx:,} 仍可完整載入的最大模型"
                    f"（active {lc.params_active / 1e9:.1f}B）"),
            caveats=list(long_caveats),
        ))
    else:
        notes.append(
            f"context {long_ctx:,}（約 {long_ctx // 1024}K）下沒有可完整"
            "載入的組合——長文需求請降低 context 或參考矩陣的 trade-off 曲線"
        )

    return RecResult(picks=picks, notes=notes)
