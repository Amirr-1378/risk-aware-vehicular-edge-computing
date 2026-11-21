"""Metadata-only compatibility and stable CPU resume identity, without Torch."""

import builtins
import contextlib
import io
import json
import tempfile
import unittest
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

from experiments import runtime_preflight as preflight


class RuntimePreflightTests(unittest.TestCase):
    def versions(self, torch="2.6.0+cpu", numpy="1.26.0"):
        values = {"torch": torch, "numpy": numpy}

        def installed(name):
            value = values[name]
            if value is None:
                raise metadata.PackageNotFoundError(name)
            return value

        return patch.object(preflight.metadata, "version", side_effect=installed)

    def collect(self, **kwargs):
        with patch.object(preflight, "source_provenance", return_value={"source_sha256": "fixture"}), \
                patch.object(preflight.sys, "version_info", (3, 12, 1)), \
                patch.object(preflight.platform, "python_version", return_value="3.12.1"):
            return preflight.collect_preflight(**kwargs)

    def test_supported_stable_versions_and_local_build_tags(self):
        for torch, numpy in (("2.6.0+cpu", "1.26.0"), ("2.14.1+cu128", "2.5.4"), ("2.6", "1.26.0.post1")):
            with self.subTest(torch=torch, numpy=numpy), self.versions(torch, numpy):
                report = self.collect(require_learning=True)
                self.assertTrue(report["ok"])
                self.assertTrue(report["ready_for_learning"])
                self.assertEqual(report["runtime_identity"]["torch_version"], torch)
                self.assertEqual(report["missing"], [])
                self.assertEqual(report["incompatible"], [])

    def test_missing_learning_dependencies_are_explicit_and_optional_in_core_mode(self):
        with self.versions(None, None):
            core = self.collect()
            required = self.collect(require_learning=True)
        self.assertEqual(core["missing"], ["torch", "numpy"])
        self.assertTrue(core["ok"])
        self.assertFalse(core["ready_for_learning"])
        self.assertFalse(required["ok"])

    def test_unsupported_or_preview_dependency_versions_do_not_claim_compatibility(self):
        for torch, numpy, wrong in (("2.5.9", "1.26.0", "torch"), ("3.0.0", "1.26.0", "torch"),
                ("2.6.0", "1.25.9", "numpy"), ("2.6.0", "3.0.0", "numpy"),
                ("2.6.0rc1", "1.26.0", "torch"), ("broken", "1.26.0", "torch")):
            with self.subTest(torch=torch, numpy=numpy), self.versions(torch, numpy):
                report = self.collect(require_learning=True)
                self.assertFalse(report["ok"])
                self.assertIn(wrong, report["incompatible"])

    def test_unsupported_python_is_required_even_in_core_mode(self):
        with self.versions(), patch.object(preflight.sys, "version_info", (3, 10, 99)), \
                patch.object(preflight.platform, "python_version", return_value="3.10.99"), \
                patch.object(preflight, "source_provenance", return_value={}):
            report = preflight.collect_preflight()
        self.assertFalse(report["ok"])
        self.assertIn("python", report["incompatible"])

    def test_identity_preserves_exact_build_and_platform_but_has_no_clock_or_hostname(self):
        with self.versions(), patch.object(preflight.platform, "system", return_value="Linux"), \
                patch.object(preflight.platform, "machine", return_value="x86_64"):
            one, two = preflight.runtime_identity(), preflight.runtime_identity()
        self.assertEqual(one, two)
        self.assertEqual(one["platform_system"], "Linux")
        self.assertEqual(one["machine"], "x86_64")
        self.assertEqual(one["device"], "cpu")
        self.assertEqual(one["torch_threads"], 1)
        self.assertNotIn("hostname", one)
        self.assertNotIn("timestamp", one)
        self.assertEqual(preflight._identity_fingerprint(one), preflight._identity_fingerprint(dict(reversed(list(one.items())))))
        self.assertNotEqual(preflight._identity_fingerprint(one), preflight._identity_fingerprint({**one, "torch_version": "2.6.0+cu128"}))

    def test_collector_never_imports_learning_libraries_or_requests_network(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "numpy", "socket", "urllib", "requests"):
                raise AssertionError("preflight must remain metadata-only")
            return original(name, *args, **kwargs)

        with self.versions(), patch("builtins.__import__", side_effect=guarded):
            report = self.collect(require_learning=True)
        self.assertTrue(report["ok"])
        self.assertFalse(report["training_started"])
        self.assertFalse(report["evaluation_started"])
        self.assertFalse(report["test_split_accessed"])

    def test_cli_serializes_exclusively_and_returns_failure_for_required_missing_libs(self):
        with tempfile.TemporaryDirectory() as directory, self.versions(None, None):
            output = Path(directory) / "preflight.json"
            with patch.object(preflight, "source_provenance", return_value={}), contextlib.redirect_stdout(io.StringIO()) as stream:
                result = preflight.main(["--require-learning", "--output", str(output)])
            self.assertEqual(result, 1)
            saved = output.read_bytes()
            self.assertFalse(json.loads(saved)["ok"])
            self.assertEqual(json.loads(stream.getvalue())["output"], str(output))
            with self.assertRaises(FileExistsError):
                preflight.main(["--output", str(output)])
            self.assertEqual(output.read_bytes(), saved)

    def test_cli_without_output_prints_finite_report_and_starts_no_execution(self):
        with self.versions(), patch.object(preflight, "source_provenance", return_value={}), contextlib.redirect_stdout(io.StringIO()) as stream:
            result = preflight.main(["--require-learning"])
        self.assertEqual(result, 0)
        report = json.loads(stream.getvalue())
        self.assertEqual(report["report_schema_version"], 1)
        self.assertEqual(report["purpose"], "development-runtime-preflight-not-training")

    def test_nonboolean_requirement_is_rejected(self):
        with self.assertRaises(TypeError):
            preflight.collect_preflight(require_learning=1)
