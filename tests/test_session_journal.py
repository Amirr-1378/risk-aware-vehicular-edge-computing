"""Dependency-free persistence transitions and corruption/recovery checks."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.session_journal import (
    append_step, copy_verified_prefix, create_session, inspect_session,
)


class SessionJournalTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "session"
        self.contract = {"purpose": "development", "seeds": (11, 29)}
        create_session(self.directory, self.contract)

    def step_path(self, index=0):
        return self.directory / "steps" / f"{index:06d}"

    @staticmethod
    def checkpoint(path):
        with path.open("xb") as stream:
            stream.write(b"checkpoint payload")

    def append(self, *, checkpoint=True):
        return append_step(self.directory, self.contract, {"completed": 1}, {"cost": 2.5},
                           self.checkpoint if checkpoint else None)

    def rewrite_commit(self, operation):
        path = self.step_path() / "commit.json"
        data = json.loads(path.read_text())
        operation(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_initial_empty_and_normal_two_step_history(self):
        self.assertEqual(inspect_session(self.directory, self.contract), [])
        first = self.append()
        second = self.append(checkpoint=False)
        entries = inspect_session(self.directory, {"seeds": [11, 29], "purpose": "development"})
        self.assertEqual(entries, [first, second])
        self.assertEqual(first["step_index"], 0)
        self.assertEqual(set(first["artifacts"]), {"report.json", "agent.pt"})
        self.assertEqual(set(second["artifacts"]), {"report.json"})
        commit = json.loads((self.step_path(1) / "commit.json").read_text())
        self.assertEqual(commit["previous_commit_sha256"], first["commit_sha256"])
        self.assertEqual(first["commit_sha256"], hashlib.sha256(
            (self.step_path() / "commit.json").read_bytes()).hexdigest())

    def test_create_and_recovery_destinations_are_exclusive(self):
        with self.assertRaises(FileExistsError):
            create_session(self.directory, self.contract)
        with self.assertRaises(FileExistsError):
            copy_verified_prefix(self.directory, self.directory, self.contract)
        with self.assertRaisesRegex(ValueError, "outside"):
            copy_verified_prefix(self.directory, self.directory / "nested-recovery", self.contract)
        self.assertEqual(inspect_session(self.directory, self.contract), [])

    def test_mismatched_contract_rejected_without_new_step(self):
        for operation in (inspect_session, copy_verified_prefix):
            with self.subTest(operation=operation.__name__), self.assertRaisesRegex(ValueError, "contract mismatch"):
                if operation is copy_verified_prefix:
                    operation(self.directory, self.root / "recovered", {"purpose": "different"})
                else:
                    operation(self.directory, {"purpose": "different"})
        with self.assertRaises(ValueError):
            append_step(self.directory, {"purpose": "different"}, {}, {})
        self.assertFalse((self.root / "recovered").exists())
        self.assertEqual(list((self.directory / "steps").iterdir()), [])

    def test_report_and_checkpoint_corruption_rejected(self):
        self.append()
        for name in ("report.json", "agent.pt"):
            path = self.step_path() / name
            original = path.read_bytes()
            path.write_bytes(b"corrupt")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "hash mismatch"):
                inspect_session(self.directory, self.contract)
            with self.assertRaises(ValueError):
                copy_verified_prefix(self.directory, self.root / "recovery", self.contract)
            self.assertFalse((self.root / "recovery").exists())
            path.write_bytes(original)

    def test_callback_error_leaves_evidence_and_requires_prefix_recovery(self):
        self.append()
        def interrupted(path):
            path.write_bytes(b"partial")
            raise RuntimeError("interrupted")
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            append_step(self.directory, self.contract, {}, {}, interrupted)
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            inspect_session(self.directory, self.contract)
        with self.assertRaises(ValueError):
            self.append()
        destination = self.root / "recovery"
        self.assertEqual(copy_verified_prefix(self.directory, destination, self.contract), 1)
        self.assertEqual(len(inspect_session(destination, self.contract)), 1)
        self.assertTrue((self.step_path(1) / "agent.pt").exists())
        self.assertFalse((self.step_path(1) / "commit.json").exists())
        append_step(destination, self.contract, {"completed": 2}, {}, self.checkpoint)
        self.assertEqual(len(inspect_session(destination, self.contract)), 2)

    def test_incomplete_initial_step_can_recover_empty_prefix(self):
        self.step_path().mkdir()
        (self.step_path() / "report.json").write_bytes(b"partial JSON {")
        with self.assertRaises(ValueError):
            inspect_session(self.directory, self.contract)
        destination = self.root / "recovery"
        self.assertEqual(copy_verified_prefix(self.directory, destination, self.contract), 0)
        self.assertEqual(inspect_session(destination, self.contract), [])

    def test_truncated_pending_final_commit_recovers_verified_prefix(self):
        self.append()
        self.step_path(1).mkdir()
        (self.step_path(1) / "report.json").write_bytes(b"{}")
        (self.step_path(1) / "commit.pending").write_bytes(b'{"schema_version":')
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            inspect_session(self.directory, self.contract)
        destination = self.root / "recovery"
        self.assertEqual(copy_verified_prefix(self.directory, destination, self.contract), 1)
        append_step(destination, self.contract, {"completed": 2}, {}, self.checkpoint)
        self.assertEqual(len(inspect_session(destination, self.contract)), 2)
        self.assertEqual((self.step_path(1) / "commit.pending").read_bytes(), b'{"schema_version":')

    def test_failure_during_commit_publication_leaves_recoverable_pending(self):
        self.append()
        with patch("experiments.session_journal.os.rename", side_effect=OSError("interrupted publication")):
            with self.assertRaisesRegex(OSError, "publication"):
                self.append()
        self.assertTrue((self.step_path(1) / "commit.pending").exists())
        self.assertFalse((self.step_path(1) / "commit.json").exists())
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            inspect_session(self.directory, self.contract)
        destination = self.root / "recovery"
        self.assertEqual(copy_verified_prefix(self.directory, destination, self.contract), 1)
        append_step(destination, self.contract, {"completed": 2}, {}, self.checkpoint)
        self.assertEqual(len(inspect_session(destination, self.contract)), 2)

    def test_published_commit_with_pending_or_malformed_json_cannot_recover(self):
        self.append()
        (self.step_path() / "commit.pending").write_bytes(b"partial")
        with self.assertRaisesRegex(ValueError, "both published and pending"):
            copy_verified_prefix(self.directory, self.root / "recovery", self.contract)
        (self.step_path() / "commit.pending").unlink()
        (self.step_path() / "commit.json").write_bytes(b"{")
        with self.assertRaises(ValueError):
            copy_verified_prefix(self.directory, self.root / "recovery", self.contract)
        self.assertFalse((self.root / "recovery").exists())

    def test_incomplete_middle_and_noncontiguous_steps_cannot_recover(self):
        self.step_path().mkdir()
        (self.step_path() / "commit.pending").write_bytes(b"partial")
        self.step_path(1).mkdir()
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            copy_verified_prefix(self.directory, self.root / "recovery", self.contract)
        self.step_path(1).rename(self.step_path(2))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            inspect_session(self.directory, self.contract)

    def test_malformed_commit_fields_schema_and_chain_rejected(self):
        self.append()
        path = self.step_path() / "commit.json"
        original = path.read_bytes()
        changes = (
            lambda value: value.update(schema_version=True),
            lambda value: value.update(step_index=True),
            lambda value: value.update(step_index=1),
            lambda value: value.update(previous_commit_sha256="0" * 64),
            lambda value: value.update(contract_sha256="0" * 64),
            lambda value: value.update(extra=1),
            lambda value: value.update(metadata=[]),
            lambda value: value.update(artifacts={"report.json": "bad"}),
            lambda value: value.update(artifacts={"../report.json": "0" * 64}),
        )
        for change in changes:
            with self.subTest(change=change):
                path.write_bytes(original)
                self.rewrite_commit(change)
                with self.assertRaises(ValueError):
                    inspect_session(self.directory, self.contract)
        path.write_bytes(b'{"schema_version":1,"schema_version":1}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            inspect_session(self.directory, self.contract)

    def test_prior_commit_modification_breaks_chain(self):
        self.append()
        self.append()
        path = self.step_path() / "commit.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "chain"):
            inspect_session(self.directory, self.contract)

    def test_nonfinite_nonobject_duplicate_or_bad_json_rejected(self):
        contract_path = self.directory / "contract.json"
        for raw in (b"[]", b"{", b'{"x":NaN}', b'{"x":1,"x":2}'):
            with self.subTest(raw=raw):
                contract_path.write_bytes(raw)
                with self.assertRaises((TypeError, ValueError)):
                    inspect_session(self.directory, self.contract)

    def test_invalid_inputs_do_not_create_output_or_partial_step(self):
        for contract in ({"x": float("nan")}, {1: "key"}, []):
            destination = self.root / "invalid"
            with self.subTest(contract=contract), self.assertRaises((TypeError, ValueError)):
                create_session(destination, contract)
            self.assertFalse(destination.exists())
        for metadata, report in (([], {}), ({}, []), ({"x": float("inf")}, {})):
            with self.assertRaises((TypeError, ValueError)):
                append_step(self.directory, self.contract, metadata, report)
            self.assertEqual(list((self.directory / "steps").iterdir()), [])
        with self.assertRaises(TypeError):
            append_step(self.directory, self.contract, {}, {}, checkpoint_writer="not callable")
        self.assertEqual(list((self.directory / "steps").iterdir()), [])

    def test_unexpected_paths_and_missing_artifacts_rejected(self):
        self.append()
        extra = self.directory / "unexpected"
        extra.write_text("x")
        with self.assertRaisesRegex(ValueError, "contents"):
            inspect_session(self.directory, self.contract)
        extra.unlink()
        extra = self.step_path() / "unexpected"
        extra.write_text("x")
        with self.assertRaisesRegex(ValueError, "contents"):
            inspect_session(self.directory, self.contract)
        extra.unlink()
        (self.step_path() / "agent.pt").unlink()
        with self.assertRaisesRegex(ValueError, "artifacts"):
            inspect_session(self.directory, self.contract)

    def test_callback_missing_or_unexpected_artifact_cannot_commit(self):
        def invalid(path):
            path.write_bytes(b"checkpoint")
            path.with_name("extra").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            append_step(self.directory, self.contract, {}, {}, invalid)
        self.assertFalse((self.step_path() / "commit.json").exists())

    def test_callback_cannot_change_report_before_commit(self):
        def invalid(path):
            path.write_bytes(b"checkpoint")
            path.with_name("report.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "changed before"):
            append_step(self.directory, self.contract, {}, {"cost": 3}, invalid)
        self.assertFalse((self.step_path() / "commit.json").exists())

    def test_symbolic_artifact_rejected_when_platform_allows(self):
        self.append()
        path = self.step_path() / "agent.pt"
        external = self.root / "external.pt"
        external.write_bytes(path.read_bytes())
        path.unlink()
        try:
            path.symlink_to(external)
        except (NotImplementedError, OSError):
            self.skipTest("symbolic links unavailable for this account")
        with self.assertRaisesRegex(ValueError, "regular"):
            inspect_session(self.directory, self.contract)


if __name__ == "__main__":
    unittest.main()
