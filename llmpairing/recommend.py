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


class RecNote(BaseModel):
    """A structured, renderer-translatable message (review #4: message
    keys, never full-sentence reverse lookup)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    params: dict[str, int | str] = {}


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
    #: review #4 P1: source trust — "official" / "trusted_quantizer" /
    #: "community". Conservative default: unknown provenance = community.
    trust: str = "community"
    #: fit/probe honesty flags for this cell (review #4 P2: the CLI must
    #: not swallow e.g. NVIDIA_SMI_PARSER_UNVERIFIED_ON_REAL_HW)
    flags: list[str] = []


class Pick(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["CAPABILITY", "SPEED", "LONG_CONTEXT"]
    candidate: RecCandidate
    reason: str
    caveats: list[str] = []  # message CODES — renderers own the words


class RecResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    picks: list[Pick] = []
    notes: list[RecNote] = []


_TRUSTED = ("official", "trusted_quantizer")


def _pool(cands: list[RecCandidate], ctx: int) -> tuple[list[RecCandidate], list[str]]:
    """Runnable candidates at ctx; TIGHT fallback carries a caveat code."""
    run = [c for c in cands if c.ctx == ctx and c.verdict in _RUNNABLE]
    if run:
        return run, []
    tight = [c for c in cands if c.ctx == ctx and c.verdict == "TIGHT"]
    if tight:
        return tight, ["TIGHT_FALLBACK"]
    return [], []


def _by_capability(c: RecCandidate) -> tuple[int, int]:
    return (c.params_active, c.weights_bytes or 0)


def recommend(cands: list[RecCandidate], *, target_ctx: int = 8_192,
              long_ctx: int = 32_768,
              include_community: bool = False) -> RecResult:
    picks: list[Pick] = []
    notes: list[RecNote] = []

    # review #4 P1: community sources are opt-in, never a silent default
    if not include_community:
        kept = [c for c in cands if c.trust in _TRUSTED]
        excluded = len({c.model_id for c in cands}) - len({c.model_id for c in kept})
        if excluded > 0:
            notes.append(RecNote(code="COMMUNITY_EXCLUDED",
                                 params={"count": excluded}))
        cands = kept

    def _caveats(base: list[str], c: RecCandidate) -> list[str]:
        out = list(base)
        if c.trust == "community":
            out.append("COMMUNITY_SOURCE")
        return out

    pool, pool_caveats = _pool(cands, target_ctx)
    if not pool:
        notes.append(RecNote(code="NO_FIT_AT_TARGET",
                             params={"ctx": target_ctx}))
        # review #5 P1: if the request itself defeated models (ctx beyond
        # their declared max), say WHY — that is actionable, a generic
        # no-fit is not
        exceeded = {c.model_id for c in cands
                    if c.ctx == target_ctx
                    and c.verdict == "CTX_EXCEEDS_MODEL_MAX"}
        if exceeded:
            notes.append(RecNote(code="CTX_EXCEEDS_MODEL_MAX",
                                 params={"ctx": target_ctx,
                                         "count": len(exceeded)}))
        return RecResult(picks=[], notes=notes)

    # -- capability: largest active params, tie -> larger weights (higher quant)
    cap = max(pool, key=_by_capability)
    picks.append(Pick(
        kind="CAPABILITY",
        candidate=cap,
        reason=(f"可完整載入的最大模型（active "
                f"{cap.params_active / 1e9:.1f}B 參數）"),
        caveats=_caveats(pool_caveats, cap),
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
            speed_caveats.append("SPEED_RELATIVE_ORDER")
    if ranked:
        distinct = [c for c in ranked if c.model_id != cap.model_id]
        speed = distinct[0] if distinct else ranked[0]
        tps_txt = (f"預估 ~{speed.tps:.0f} tok/s" if speed.tps is not None
                   else "每 token 記憶體流量最低")
        picks.append(Pick(
            kind="SPEED", candidate=speed,
            reason=f"可完整載入組合中最快（{tps_txt}）",
            caveats=_caveats(speed_caveats, speed),
        ))

    # -- long context: capability ranking at the long ctx point
    long_pool, long_caveats = _pool(cands, long_ctx)
    if long_pool:
        lc = max(long_pool, key=_by_capability)
        picks.append(Pick(
            kind="LONG_CONTEXT", candidate=lc,
            reason=(f"在 context {long_ctx:,} 仍可完整載入的最大模型"
                    f"（active {lc.params_active / 1e9:.1f}B）"),
            caveats=_caveats(long_caveats, lc),
        ))
    else:
        notes.append(RecNote(code="NO_FIT_AT_LONG", params={"ctx": long_ctx}))

    return RecResult(picks=picks, notes=notes)
