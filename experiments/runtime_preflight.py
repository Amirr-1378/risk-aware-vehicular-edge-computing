"""Metadata-only CPU runtime preflight; no install, training or evaluation.

Torch and NumPy are deliberately not imported. Version metadata checks do not
prove library importability or cross-platform bitwise reproducibility. The
execution settings in the identity are the required CPU driver contract, not
a claim that this collector changed process settings.
"""

import argparse
import hashlib
import json
import platform
import re
import sys
from importlib import metadata
from pathlib import Path

from risk_aware_v2x.contracts import model_contract
from .provenance import source_provenance


def _installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def runtime_identity() -> dict:
    """Exact environment/build identity plus the unchanged CPU execution contract.

    No host name, temporary path or wall clock enters this stable identity.
    ``python_full_version`` includes build/compiler details; exact Torch local
    build tags such as ``+cpu`` or ``+cu128`` are retained for resume guards.
    """
    return {"python_version": platform.python_version(), "python_full_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(), "machine": platform.machine(),
        "torch_version": _installed_version("torch"), "numpy_version": _installed_version("numpy"),
        "device": "cpu", "torch_threads": 1, "deterministic_algorithms": True}


def _compatible_release(value, *, minimum):
    """Check stable release metadata, retaining local wheel tags in the identity.

    Preview/dev builds are rejected explicitly instead of treating e.g.
    2.6.0rc1 as the stable >=2.6 lower bound. Post releases are supported.
    """
    if not isinstance(value, str):
        return False
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?(?:\.post\d+)?(?:\+[A-Za-z0-9._-]+)?", value)
    return bool(match and minimum <= tuple(int(match.group(i) or 0) for i in (1, 2, 3)) < (3, 0, 0))


def _identity_fingerprint(identity):
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collect_preflight(*, require_learning=False) -> dict:
    """Collect compatibility evidence without importing learning libraries.

    Python is always required. Optional learning failures remain explicit even
    in core-only mode, but only gate ``ok`` when ``require_learning=True``.
    The caller must still exercise real imports before using the learner.
    """
    if not isinstance(require_learning, bool):
        raise TypeError("require_learning must be boolean")
    identity = runtime_identity()
    python_ok = tuple(sys.version_info[:2]) >= (3, 11)
    checks = [
        {"name": "python", "requirement": ">=3.11", "installed_version": identity["python_version"],
         "required": True, "compatible": python_ok},
        {"name": "torch", "requirement": ">=2.6,<3; stable release", "installed_version": identity["torch_version"],
         "required": require_learning, "compatible": _compatible_release(identity["torch_version"], minimum=(2, 6, 0))},
        {"name": "numpy", "requirement": ">=1.26,<3; stable release", "installed_version": identity["numpy_version"],
         "required": require_learning, "compatible": _compatible_release(identity["numpy_version"], minimum=(1, 26, 0))},
    ]
    missing = [check["name"] for check in checks if check["installed_version"] is None]
    incompatible = [check["name"] for check in checks if check["installed_version"] is not None and not check["compatible"]]
    ready = all(check["compatible"] for check in checks)
    return {"report_schema_version": 1, "purpose": "development-runtime-preflight-not-training",
        "model_contract": model_contract(), "provenance": source_provenance(),
        "runtime_identity": identity, "runtime_identity_sha256": _identity_fingerprint(identity),
        "device": "cpu", "require_learning": require_learning, "checks": checks,
        "missing": missing, "incompatible": incompatible,
        "ready_for_learning": ready, "ok": python_ok and (ready or not require_learning),
        "training_started": False, "evaluation_started": False, "test_split_accessed": False,
        "limits": ["Metadata compatibility only; learning-library imports are not attempted.",
                   "Exact resume identity does not promise cross-platform or cross-release bitwise agreement.",
                   "CPU execution is required; a Colab GPU does not automatically accelerate this driver."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-learning", action="store_true")
    parser.add_argument("--output", type=Path, help="new JSON path; existing reports are never replaced")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        raise FileExistsError("choose a new preflight output path")
    report = collect_preflight(require_learning=args.require_learning)
    if args.output is not None:
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as destination:
            destination.write(payload)
        print(json.dumps({"output": str(args.output), "ok": report["ok"],
                          "ready_for_learning": report["ready_for_learning"]}))
    else:
        print(json.dumps(report, allow_nan=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
