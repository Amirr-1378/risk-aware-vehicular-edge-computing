"""Validate and extract a browser-downloaded private GitHub source ZIP offline.

The archive root declares the expected commit; it does not cryptographically
prove Git history. A separately trusted SHA256 verifies the Python source set
using the same text normalization as ``source_provenance``. No credentials,
network requests, dependency installation or training are involved.
"""

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


_MAX_EXTRACTED_BYTES = 100 * 1024 * 1024
_REQUIRED_FILES = frozenset({"README.md", "AGENTS.md", "pyproject.toml",
    "src/risk_aware_v2x/contracts.py", "experiments/provenance.py"})
_WINDOWS_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
    *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))})


def _member_path(info, expected_root):
    name = info.filename
    if "\x00" in info.orig_filename or "\\" in info.orig_filename or name.startswith("/"):
        raise ValueError("archive contains an unsafe path")
    pieces = name.rstrip("/").split("/")
    if any(not piece or piece in (".", "..") or any(char in '<>:"|?*' or ord(char) < 32 for char in piece)
           or piece.rstrip(" .") != piece
           or piece.split(".")[0].upper() in _WINDOWS_RESERVED for piece in pieces):
        raise ValueError("archive contains an unsafe or nonportable path")
    if pieces[0] != expected_root:
        raise ValueError("archive must have exactly the expected commit root")
    if any(piece.casefold() == ".git" for piece in pieces[1:]):
        raise ValueError("archive must not contain a .git tree")
    mode_type = stat.S_IFMT((info.external_attr >> 16) & 0xFFFF)
    allowed_type = stat.S_IFDIR if info.is_dir() else stat.S_IFREG
    if mode_type not in (0, allowed_type):
        raise ValueError("archive members must be regular files or directories")
    if info.flag_bits & 1:
        raise ValueError("encrypted archive members are unsupported")
    if len(pieces) == 1 and not info.is_dir():
        raise ValueError("archive root must be a directory")
    if info.is_dir() and info.file_size:
        raise ValueError("archive directory must not contain file data")
    return PurePosixPath(*pieces[1:]).as_posix() if len(pieces) > 1 else ""


def _source_sha256(files):
    digest = hashlib.sha256()
    for relative in sorted(files):
        parts = PurePosixPath(relative).parts
        is_source = relative.endswith(".py") and (
            parts[0] == "src" or (parts[0] == "experiments" and len(parts) == 2))
        if is_source:
            # Path.read_text(encoding='utf-8') uses universal newline handling.
            text = files[relative].decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(text.encode("utf-8") + b"\0")
    return digest.hexdigest()


def extract_source_archive(archive: Path, destination: Path, *, expected_commit: str,
                           expected_source_sha256: str) -> Path:
    """Validate every member and source digest before creating a new root.

    The returned root is ``destination`` with the single GitHub archive prefix
    stripped. Existing paths are never replaced. The 100 MiB limit applies to
    the sum of uncompressed member sizes. Non-Python contents are presence/
    path checked, but are outside the source-provenance SHA256 contract.
    """
    if not isinstance(expected_commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", expected_commit):
        raise ValueError("expected_commit must be a full 40-hex commit")
    if not isinstance(expected_source_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_source_sha256):
        raise ValueError("expected_source_sha256 must be a trusted 64-hex digest")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("choose a new source destination")
    expected_root = "risk-aware-vehicular-edge-computing-" + expected_commit
    files, directories, seen, spellings = {}, set(), set(), {}
    with ZipFile(archive) as source:
        members = source.infolist()
        if sum(member.file_size for member in members) > _MAX_EXTRACTED_BYTES:
            raise ValueError("archive exceeds the 100 MiB extracted-size limit")
        for member in members:
            relative = _member_path(member, expected_root)
            portable_key = relative.casefold()
            if portable_key in seen:
                raise ValueError("archive contains duplicate or case-colliding members")
            seen.add(portable_key)
            parts = PurePosixPath(relative).parts
            for count in range(1, len(parts) + 1):
                prefix = "/".join(parts[:count])
                key = prefix.casefold()
                if key in spellings and spellings[key] != prefix:
                    raise ValueError("archive contains case-colliding directory paths")
                spellings[key] = prefix
            if member.is_dir():
                if relative:
                    directories.add(relative)
            else:
                # Reading all bytes also checks CRC/decompression before any writes.
                files[relative] = source.read(member)
        file_keys = {relative.casefold() for relative in files}
        for relative in (*files, *directories):
            for parent in PurePosixPath(relative).parents:
                if parent.as_posix().casefold() in file_keys:
                    raise ValueError("archive file conflicts with a directory")
        if not _REQUIRED_FILES.issubset(files):
            raise ValueError("archive is missing required repository files")
        if _source_sha256(files) != expected_source_sha256.lower():
            raise ValueError("archive Python sources do not match the trusted source digest")

    destination.mkdir(parents=True, exist_ok=False)
    for relative in sorted(directories, key=lambda name: (name.count("/"), name)):
        destination.joinpath(*PurePosixPath(relative).parts).mkdir(parents=True, exist_ok=True)
    for relative, data in files.items():
        target = destination.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(data)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args(argv)
    root = extract_source_archive(args.archive, args.destination,
        expected_commit=args.commit, expected_source_sha256=args.source_sha256)
    print(json.dumps({"root": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
