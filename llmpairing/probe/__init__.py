"""Hardware probe (T-001) — the dirty I/O layer.

Read-only, no elevation, no network. Its sole job is turning the outside
world into a schema-validated HardwareProfile; every subprocess goes
through runner.run_readonly (R-T1-2) and everything unobtainable is
reported as UNKNOWN, never guessed (R-T1-4).
"""
