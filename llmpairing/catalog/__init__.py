"""Model catalog (T-005) — normalization from upstream metadata to ModelSpec.

The mapper is pure (testable against archived configs); network fetching
lives in tools/catalog/ and runs on the owner's machine. Catalog output is
a VERSIONED SNAPSHOT with sha256 — never a live lookup (reproducibility
over freshness; F-11).
"""
