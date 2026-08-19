"""Build the PUBLIC branch: a whitelist export of HEAD with clean history.

The working repository (master) contains the spec pack, the test suite
and the owner's machine data — none of which is published. This script
copies only the public whitelist into the orphan branch `public`, which
is what gets force-pushed to GitHub (`git push --force origin
public:master`). Run from the repo root on a clean tree.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: everything published, nothing else. Directories are copied whole.
WHITELIST = [
    "llmpairing",
    "tools",
    "catalog",
    "README.md",
    "LICENSE",
    "pyproject.toml",
    ".gitignore",
    ".gitattributes",
]


def _git(*args: str, cwd: Path = REPO) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def main() -> int:
    head = _git("rev-parse", "--short", "HEAD")
    if _git("status", "--porcelain"):
        raise SystemExit("working tree not clean — commit first")

    wt = Path(tempfile.mkdtemp(prefix="llmp-public-"))
    wt.rmdir()  # git worktree add wants to create it
    try:
        if _git("branch", "--list", "public"):
            _git("worktree", "add", str(wt), "public")
        else:
            _git("worktree", "add", "--detach", str(wt), "HEAD")
            _git("checkout", "--orphan", "public", cwd=wt)
        # empty the worktree (keep .git)
        for child in wt.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        for name in WHITELIST:
            src = REPO / name
            if not src.exists():
                raise SystemExit(f"whitelist entry missing: {name}")
            dst = wt / name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        # __pycache__ never ships
        for pyc in wt.rglob("__pycache__"):
            shutil.rmtree(pyc)
        # GitHub Pages showcase: reference-machines-only demo (no personal
        # profile, no calibration effects) served from /docs on the public
        # branch -> https://<user>.github.io/<repo>/
        r = subprocess.run(
            [sys.executable, str(REPO / "tools" / "demo" / "build_demo.py"),
             "--no-profile", "--out", str(wt / "docs" / "index.html")],
            cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"showcase demo build failed: {r.stderr.strip()}")
        print(r.stdout.strip())
        _git("add", "-A", cwd=wt)
        if not _git("status", "--porcelain", cwd=wt):
            print("public branch already up to date")
            return 0
        _git("commit", "-m", f"public export of {head}", cwd=wt)
        print(f"public branch updated from {head}: "
              + _git("rev-parse", "--short", "public"))
        return 0
    finally:
        _git("worktree", "remove", "--force", str(wt))


if __name__ == "__main__":
    raise SystemExit(main())
