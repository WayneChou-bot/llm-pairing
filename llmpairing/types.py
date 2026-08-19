"""Frozen core types for LLM pairing (T-002 SPEC §3).

S1 scope: data structures only — zero computation code.
"""
from __future__ import annotations

from enum import Enum
from typing import Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict

#: KV cache dtype accepted by llama.cpp (SPEC §2.3 / §4.3). Single source of
#: truth shared by Workload and budget/kv.py.
KvDtype: TypeAlias = Literal["f16", "q8_0", "q4_0"]


class Tier(str, Enum):
    """Evidence tier. T0 = estimated from spec DB, T1 = probe-calibrated, T2 = fully measured.

    SPEC §3.1 propagation rule: an output's tier is the MINIMUM of its input tiers.
    Values are chosen so that plain ``min()`` (str ordering: "T0" < "T1" < "T2")
    yields exactly that semantics; test_tier_ordering_supports_min_propagation pins it.
    Direct construction of ``tier=`` outside TierTracker is forbidden from S8 onward (R-3).
    """

    T0 = "T0"
    T1 = "T1"
    T2 = "T2"


class Source(str, Enum):
    """Provenance of a numeric input (SPEC §2.1 / §2.2)."""

    SPEC_DB = "SPEC_DB"
    MEASURED = "MEASURED"
    UNKNOWN = "UNKNOWN"
    CATALOG_VERIFIED = "CATALOG_VERIFIED"
    ESTIMATED = "ESTIMATED"


class Verdict(str, Enum):
    """Fit classification (SPEC §3.2). Declaration order IS severity order.

    No UNKNOWN / PROBABLY_OK style members may ever be added (SPEC §3.2, R-2):
    when we don't know, we answer UNSUPPORTED_*.
    """

    FITS = "FITS"
    TIGHT = "TIGHT"
    OOM_AT_CONTEXT = "OOM_AT_CONTEXT"
    PARTIAL_OFFLOAD = "PARTIAL_OFFLOAD"
    RAM_ONLY = "RAM_ONLY"
    OOM_AT_LOAD = "OOM_AT_LOAD"
    UNSUPPORTED_ARCH = "UNSUPPORTED_ARCH"
    UNSUPPORTED_TOPOLOGY = "UNSUPPORTED_TOPOLOGY"
    UNSUPPORTED_ENGINE = "UNSUPPORTED_ENGINE"


T = TypeVar("T")


class Measured(BaseModel, Generic[T]):
    """A value that carries its own evidence level (SPEC §3.1).

    ``value is None`` is a CORRECT output, not a failure (R-2): it means
    "we refuse to guess". ``ci_low`` / ``ci_high`` are only meaningful for T1/T2.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: T | None
    unit: str
    tier: Tier
    source: Source
    ci_low: T | None = None
    ci_high: T | None = None
    notes: list[str] = []


class DemandBreakdown(BaseModel):
    """Memory demand split into its four mandatory components (SPEC §4.2, P-11).

    All four fields are required — omitting ``logits`` is exactly the bug
    P-11 exists to prevent. V-P1 asserts the components sum to the total.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    weights: Measured[int]
    kv_cache: Measured[int]
    activation: Measured[int]
    logits: Measured[int]
