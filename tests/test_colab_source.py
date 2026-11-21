"""Offline private-source import rejects unsafe ZIPs before making a root."""

import contextlib
import hashlib
import io
import json
import stat
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

from experiments import colab_source


class ColabSourceTests(unittest.TestCase):
    commit = "a" * 40
    root = "risk-aware-vehicular-edge-computing-" + commit

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.archive = self.base / "user-upload-with-any-name.zip"
        self.destination = self.base / "source"
        self.files = {"README.md": b"readme", "AGENTS.md": b"guidance", "pyproject.toml": b"[project]\n",
            "src/risk_aware_v2x/contracts.py": b"CONTRACT = 1\r\n",
            "src/risk_aware_v2x/entities/task.py": b"TASK = 1\r",
            "experiments/provenance.py": b"PROVENANCE = 1\n",
            "experiments/nested/not_fingerprinted.py": b"# outside provenance scope\n"}

    def digest(self):
        value = hashlib.sha256()
        for path in sorted(self.files):
            if path.endswith(".py") and (path.startswith("src/") or (
                    path.startswith("experiments/") and path.count("/") == 1)):
                value.update(path.encode() + b"\0")
                value.update(self.files[path].decode().replace("\r\n", "\n").replace("\r", "\n").encode() + b"\0")
        return value.hexdigest()

    def write(self, extra=(), *, root=None):
        with ZipFile(self.archive, "w") as output, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            prefix = self.root if root is None else root
            output.writestr(prefix + "/", b"")
            for name, value in self.files.items():
                output.writestr(prefix + "/" + name, value)
            for name, value in extra:
                output.writestr(name, value)

    def extract(self, **overrides):
        options = {"expected_commit": self.commit, "expected_source_sha256": self.digest(), **overrides}
        return colab_source.extract_source_archive(self.archive, self.destination, **options)

    def test_valid_archive_strips_root_and_matches_read_text_provenance(self):
        self.write()
        result = self.extract()
        self.assertEqual(result, self.destination)
        self.assertFalse((result / self.root).exists())
        value = hashlib.sha256()
        paths = sorted([*(result / "src").rglob("*.py"), *(result / "experiments").glob("*.py")])
        for path in paths:
            value.update(path.relative_to(result).as_posix().encode() + b"\0")
            value.update(path.read_text(encoding="utf-8").encode() + b"\0")
        self.assertEqual(value.hexdigest(), self.digest())
        self.assertEqual((result / "README.md").read_bytes(), b"readme")

    def test_bad_expected_identities_and_source_digest_leave_no_output(self):
        self.write()
        for values in ({"expected_commit": "main"}, {"expected_commit": "b" * 40},
                       {"expected_source_sha256": "short"}, {"expected_source_sha256": "0" * 64}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.extract(**values)
            self.assertFalse(self.destination.exists())

    def test_wrong_root_and_prefix_sibling_leave_no_output(self):
        for root in (self.root + "-sibling", "other-" + self.commit):
            self.write(root=root)
            with self.subTest(root=root), self.assertRaises(ValueError):
                self.extract()
            self.assertFalse(self.destination.exists())

    def test_unsafe_paths_and_git_trees_leave_no_output(self):
        for name in ("../outside", "/absolute", "C:/outside", self.root + "/../outside",
                     self.root + "/.git/config", self.root + "/.GIT/config",
                     self.root + "/.git./config", self.root + "/NUL", self.root + "/a//b",
                     self.root + "/bad*name", self.root + '/bad"name'):
            self.write([(name, b"bad")])
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.extract()
            self.assertFalse(self.destination.exists())

    def test_backslash_encoded_in_zip_is_rejected_before_windows_normalizes_it(self):
        self.write([(self.root + "/a/outside", b"bad")])
        self.archive.write_bytes(self.archive.read_bytes().replace(b"/a/outside", b"/a\\outside"))
        with self.assertRaises(ValueError):
            self.extract()
        self.assertFalse(self.destination.exists())

    def test_duplicate_and_case_collision_leave_no_output(self):
        for name in (self.root + "/README.md", self.root + "/readme.md"):
            self.write([(name, b"bad")])
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.extract()
            self.assertFalse(self.destination.exists())

    def test_implicit_directory_case_aliases_leave_no_output(self):
        self.write([(self.root + "/folder/one", b"one"), (self.root + "/FOLDER/two", b"two")])
        with self.assertRaises(ValueError):
            self.extract()
        self.assertFalse(self.destination.exists())

    def test_symlinks_and_special_files_leave_no_output(self):
        for mode in (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFSOCK, stat.S_IFDIR):
            info = ZipInfo(self.root + "/unsafe")
            info.create_system = 3
            info.external_attr = (mode | 0o777) << 16
            self.write([(info, b"target")])
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.extract()
            self.assertFalse(self.destination.exists())

    def test_file_directory_conflict_leaves_no_output(self):
        self.write([(self.root + "/conflict", b"file"), (self.root + "/conflict/child", b"child")])
        with self.assertRaises(ValueError):
            self.extract()
        self.assertFalse(self.destination.exists())

    def test_required_files_and_utf8_are_checked_before_writes(self):
        del self.files["pyproject.toml"]
        self.write()
        with self.assertRaises(ValueError):
            self.extract()
        self.assertFalse(self.destination.exists())
        self.files["pyproject.toml"] = b"[project]\n"
        self.files["experiments/provenance.py"] = b"\xff"
        self.write()
        with self.assertRaises(UnicodeDecodeError):
            self.extract(expected_source_sha256="0" * 64)
        self.assertFalse(self.destination.exists())

    def test_extracted_size_limit_is_enforced_before_writes(self):
        self.write()
        with patch.object(colab_source, "_MAX_EXTRACTED_BYTES", 10), self.assertRaises(ValueError):
            self.extract()
        self.assertFalse(self.destination.exists())

    def test_encrypted_member_is_rejected_before_writes(self):
        self.write()
        data = bytearray(self.archive.read_bytes())
        local = data.index(b"PK\x03\x04")
        central = data.index(b"PK\x01\x02")
        data[local + 6] |= 1
        data[central + 8] |= 1
        self.archive.write_bytes(data)
        with self.assertRaises(ValueError):
            self.extract()
        self.assertFalse(self.destination.exists())

    def test_existing_destination_is_never_changed(self):
        self.write()
        self.destination.mkdir()
        marker = self.destination / "keep"
        marker.write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            self.extract()
        self.assertEqual(marker.read_bytes(), b"original")
        self.assertEqual(list(self.destination.iterdir()), [marker])

    def test_cli_reports_root_without_network_or_training(self):
        self.write()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = colab_source.main(["--archive", str(self.archive), "--destination", str(self.destination),
                "--commit", self.commit, "--source-sha256", self.digest()])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue()), {"root": str(self.destination)})
