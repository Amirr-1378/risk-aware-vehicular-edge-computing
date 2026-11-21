"""Immutable episode-boundary artifacts for a single-writer development run.

A commit is written to ``commit.pending`` and renamed last in each numbered
step. Hashes detect incomplete or changed artifacts; they do not make
Drive/FUSE storage a transactional database.
There is no distributed lock or promise of remote durability. A failed write is
retained for inspection and requires recovery into a new directory.
"""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Callable


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_STEP = re.compile(r"[0-9]{6}\Z")
_ARTIFACTS = {"report.json", "agent.pt"}


def _canonical(value: dict) -> bytes:
    if not isinstance(value, dict):
        raise TypeError("journal records must be dictionaries")
    # JSON normalization deliberately treats tuples and lists alike.
    def validate_keys(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("JSON object keys must be strings")
            for child in item.values():
                validate_keys(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                validate_keys(child)
    validate_keys(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_json(path: Path) -> tuple[dict, bytes]:
    payload = path.read_bytes()
    record = json.loads(payload, object_pairs_hook=_unique, parse_constant=_reject_constant)
    _canonical(record)
    return record, payload


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _plain_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular journal file: {path.name}")


def _plain_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"expected a regular journal directory: {path.name}")


def create_session(directory: str | Path, contract: dict) -> None:
    """Create a new journal exclusively; never replace an existing directory."""
    payload = _canonical(contract)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    _write(directory / "contract.json", payload)
    (directory / "steps").mkdir()


def _scan(directory: Path, contract: dict, *, allow_incomplete_final: bool) -> list[dict]:
    expected = _canonical(contract)
    _plain_directory(directory)
    if {path.name for path in directory.iterdir()} != {"contract.json", "steps"}:
        raise ValueError("unexpected or missing session contents")
    contract_path, steps_path = directory / "contract.json", directory / "steps"
    _plain_file(contract_path)
    stored, _ = _read_json(contract_path)
    if _canonical(stored) != expected:
        raise ValueError("session contract mismatch")
    _plain_directory(steps_path)
    paths = sorted(steps_path.iterdir(), key=lambda path: path.name)
    if any(_STEP.fullmatch(path.name) is None for path in paths):
        raise ValueError("unexpected step path")
    if [path.name for path in paths] != [f"{index:06d}" for index in range(len(paths))]:
        raise ValueError("session steps must form a contiguous prefix")
    entries = []
    previous = None
    for index, path in enumerate(paths):
        _plain_directory(path)
        files = {child.name: child for child in path.iterdir()}
        if not set(files) <= _ARTIFACTS | {"commit.json", "commit.pending"}:
            raise ValueError("unexpected step contents")
        for child in files.values():
            _plain_file(child)
        if "commit.json" in files and "commit.pending" in files:
            raise ValueError("ambiguous step has both published and pending commits")
        if "commit.json" not in files:
            if allow_incomplete_final and index == len(paths) - 1:
                break
            raise ValueError("uncommitted step; recover verified prefix into a new directory")
        commit, payload = _read_json(files["commit.json"])
        if set(commit) != {"schema_version", "step_index", "contract_sha256",
                           "previous_commit_sha256", "metadata", "artifacts"}:
            raise ValueError("invalid step commit fields")
        if (type(commit["schema_version"]) is not int or commit["schema_version"] != 1
                or type(commit["step_index"]) is not int or commit["step_index"] != index
                or commit["contract_sha256"] != _sha(expected)
                or commit["previous_commit_sha256"] != previous):
            raise ValueError("step commit chain or contract mismatch")
        if not isinstance(commit["metadata"], dict) or not isinstance(commit["artifacts"], dict):
            raise ValueError("invalid step metadata or artifacts")
        artifacts = commit["artifacts"]
        if set(artifacts) not in ({"report.json"}, _ARTIFACTS):
            raise ValueError("invalid artifact set")
        if set(files) != set(artifacts) | {"commit.json"}:
            raise ValueError("step artifacts do not match committed files")
        for name, expected_sha in artifacts.items():
            if not isinstance(expected_sha, str) or _DIGEST.fullmatch(expected_sha) is None:
                raise ValueError("invalid artifact digest")
            if _file_sha(files[name]) != expected_sha:
                raise ValueError(f"artifact hash mismatch: {name}")
        _read_json(files["report.json"])
        previous = _sha(payload)
        entries.append({"step_index": index, "metadata": commit["metadata"],
                        "artifacts": artifacts, "commit_sha256": previous})
    return entries


def inspect_session(directory: str | Path, contract: dict) -> list[dict]:
    """Verify every artifact and commit; fail closed on any incomplete step."""
    return _scan(Path(directory), contract, allow_incomplete_final=False)


def append_step(directory: str | Path, contract: dict, metadata: dict, report: dict,
                checkpoint_writer: Callable[[Path], None] | None = None) -> dict:
    """Write immutable artifacts, then publish their commit as the final write.

    The callback must exclusively create ``agent.pt`` and return after closing
    it. Any exception leaves evidence in place; this function never deletes it.
    """
    report_payload = _canonical(report)
    metadata_payload = _canonical(metadata)
    if checkpoint_writer is not None and not callable(checkpoint_writer):
        raise TypeError("checkpoint_writer must be callable or None")
    directory = Path(directory)
    entries = inspect_session(directory, contract)
    index = len(entries)
    if index >= 1_000_000:
        raise ValueError("journal step limit exceeded")
    path = directory / "steps" / f"{index:06d}"
    path.mkdir(exist_ok=False)
    _write(path / "report.json", report_payload)
    artifacts = {"report.json": _file_sha(path / "report.json")}
    if checkpoint_writer is not None:
        checkpoint_writer(path / "agent.pt")
        _plain_file(path / "agent.pt")
        artifacts["agent.pt"] = _file_sha(path / "agent.pt")
    commit = {"schema_version": 1, "step_index": index,
              "contract_sha256": _sha(_canonical(contract)),
              "previous_commit_sha256": entries[-1]["commit_sha256"] if entries else None,
              "metadata": json.loads(metadata_payload), "artifacts": artifacts}
    # Ensure the callback did not introduce an unexpected or symbolic path.
    if {child.name for child in path.iterdir()} != set(artifacts):
        raise ValueError("unexpected checkpoint callback output")
    for name in artifacts:
        _plain_file(path / name)
        if _file_sha(path / name) != artifacts[name]:
            raise ValueError("artifact changed before commit publication")
    payload = _canonical(commit)
    pending, published = path / "commit.pending", path / "commit.json"
    _write(pending, payload)
    # Single-writer contract: never intentionally replace a published commit.
    # Rename is an atomic publication boundary on ordinary local filesystems;
    # it cannot promise atomic remote durability through Drive/FUSE.
    if published.exists() or published.is_symlink():
        raise FileExistsError("step commit already exists")
    os.rename(pending, published)
    return {"step_index": index, "metadata": commit["metadata"],
            "artifacts": artifacts, "commit_sha256": _sha(payload)}


def copy_verified_prefix(source: str | Path, destination: str | Path, contract: dict) -> int:
    """Recover into a NEW directory, omitting only an incomplete final step.

    Source evidence is retained. Corrupt committed steps, an incomplete middle
    step, and any unexpected/symbolic path are errors, never silently skipped.
    The source and destination must not be used by another writer during copy.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("recovery destination already exists")
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("recovery destination must be outside the source directory")
    entries = _scan(source, contract, allow_incomplete_final=True)
    create_session(destination, contract)
    for entry in entries:
        name = f"{entry['step_index']:06d}"
        old, new = source / "steps" / name, destination / "steps" / name
        new.mkdir()
        # Preserve original commit bytes so the immutable hash chain is exact.
        for filename in [*entry["artifacts"], "commit.json"]:
            with (old / filename).open("rb") as reader, (new / filename).open("xb") as writer:
                shutil.copyfileobj(reader, writer)
                writer.flush()
                os.fsync(writer.fileno())
    inspect_session(destination, contract)
    return len(entries)
