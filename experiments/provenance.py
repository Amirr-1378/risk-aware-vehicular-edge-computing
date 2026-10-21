"""Dependency-free provenance shared by analytical and neural experiments."""

import hashlib
import subprocess
from pathlib import Path


def source_provenance() -> dict:
    """Fingerprint actual Python sources, including uncommitted implementation."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = sorted([*(root / "src").rglob("*.py"), *(root / "experiments").glob("*.py")])
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_text(encoding="utf-8").encode("utf-8") + b"\0")
    commit, dirty = None, None
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=5, check=False)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=5, check=False)
        if revision.returncode == 0:
            commit = revision.stdout.strip()
        if status.returncode == 0:
            dirty = bool(status.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"source_sha256": digest.hexdigest(), "git_commit": commit, "git_dirty": dirty}
