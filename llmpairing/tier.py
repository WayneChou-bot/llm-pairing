"""Tier propagation (SPEC §3.1).

Introduced early (S3) rather than at S8 so that every Measured construction
point goes through TierTracker from day one (R-3). S8 adds the CI lint and
the injection property tests (V-P6 / V-P7).
"""
from __future__ import annotations

from typing import Any

from llmpairing.types import Measured, Tier


class TierTracker:
    """The only legitimate producer of a Measured's tier.

    Iron rule (SPEC §3.1): an output's tier is the MINIMUM of all of its
    inputs' tiers. If bandwidth is SPEC_DB (T0), the resulting tok/s is T0
    even when everything else was measured.
    """

    @staticmethod
    def combine(*inputs: "Measured[Any] | Tier") -> Tier:
        if not inputs:
            raise ValueError("TierTracker.combine requires at least one input")
        tiers = [i.tier if isinstance(i, Measured) else i for i in inputs]
        return min(tiers)  # str-enum ordering "T0" < "T1" < "T2" is pinned by test
