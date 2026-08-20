"""Stable, privacy-preserving machine fingerprint (review #4 P1).

A calibration file claims "measured on THIS machine" — that claim must
be verifiable. The fingerprint is a one-way hash of stable hardware
facts (hostname + CPU model string + total RAM); the raw values never
leave the machine and cannot be recovered from the 12-hex digest.
"""
from __future__ import annotations

import hashlib


def machine_fingerprint(hostname: str, cpu_model: str, total_bytes: int) -> str:
    """sha256(hostname|cpu|ram)[:12] — stable across reboots, opaque."""
    raw = f"{hostname}|{cpu_model}|{total_bytes}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]
