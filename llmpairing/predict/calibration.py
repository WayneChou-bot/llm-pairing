"""Machine calibration (T-003 MVP) — the measured BW x eta product.

See docs/T-003_NOTE.md for the design and honesty rules. The product is
measured by tools/calibrate/run_calibration.py on the owner's machine and
consumed by predict_decode; predictions made through it are T1.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
