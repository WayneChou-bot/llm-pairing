"""Machine calibration (T-003 MVP) — the measured BW x eta product.

See docs/T-003_NOTE.md for the design and honesty rules. The product is
measured by tools/calibrate/run_calibration.py on the owner's machine and
consumed by predict_decode; predictions made through it are T1.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llmpairing.schemas import HardwareProfile


class MachineCalibration(BaseModel):
    """One machine+pool's measured decode-throughput product with provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"]
    machine_id: str
    #: which memory pool the measurement exercised — strict matching, rule 3
    pool: Literal["cpu", "gpu"]
    backend: str                       # e.g. "ollama"
    product_bytes_per_s: float = Field(gt=0)   # median(tps_i x bpt_i)
    product_ci_low: float = Field(gt=0)        # min over runs (n=3: no fake IQR)
    product_ci_high: float = Field(gt=0)
    n_runs: int = Field(ge=1)
    calibration_model: str             # whitelisted-arch model id (rule 1)
    calibration_quant: str
    calibration_ctx: int = Field(ge=1)  # prompt + gen/2 approximation (rule 5)
    measured_at_unix: int = Field(ge=0)
    notes: list[str] = []


def calibration_applies(hw: HardwareProfile, cal: "MachineCalibration",
                        scenario_pool: str) -> tuple[bool, str | None]:
    """Single gate (review #4 P1): a calibration applies ONLY when the
    pool matches AND the profile's machine fingerprint matches the
    calibration's machine_id. Returns (applies, refusal_flag)."""
    if cal.pool != scenario_pool:
        return False, "CALIBRATION_POOL_MISMATCH_IGNORED"
    if not hw.machine_id:
        return False, "CALIBRATION_PROFILE_HAS_NO_MACHINE_ID"
    if cal.machine_id != hw.machine_id:
        return False, "CALIBRATION_MACHINE_MISMATCH_IGNORED"
    return True, None


def pick_calibration(hw: HardwareProfile,
                     cals: list["MachineCalibration"],
                     ) -> "MachineCalibration | None":
    """Choose which calibration file to hand to predict_decode.

    Field catch 2026-08-20: a stale pre-fingerprint file sorted after
    the fresh one and won the naive newest-by-name pick — the identity
    gate then (correctly) refused it, silently costing T1. Prefer the
    file whose machine_id matches the profile; with no match, return
    the last one so the gate can refuse with a visible flag.
    """
    if not cals:
        return None
    if hw.machine_id:
        for cal in reversed(cals):
            if cal.machine_id == hw.machine_id:
                return cal
    return cals[-1]
