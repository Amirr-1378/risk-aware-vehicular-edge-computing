"""Offline checks for the Colab entry point; never mount Drive or train."""

import ast
import json
import sys
import types
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_development.ipynb"


def load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells():
    return {cell["id"]: "".join(cell["source"])
            for cell in load_notebook()["cells"] if cell["cell_type"] == "code"}


class ColabNotebookTests(unittest.TestCase):
    def test_unexecuted_nbformat_four_and_compilable_python(self):
        notebook = load_notebook()
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(notebook["nbformat_minor"], 5)
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "python3")
        for identifier, source in code_cells().items():
            with self.subTest(cell=identifier):
                compile(source, f"{NOTEBOOK.name}:{identifier}", "exec")
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_defaults_require_explicit_actions_and_source_identity(self):
        namespace = {}
        exec(compile(code_cells()["configuration"], "configuration", "exec"), namespace)
        for flag in ("RUN_TESTS", "MOUNT_DRIVE", "RUN_DEVELOPMENT", "RESUME"):
            self.assertIs(namespace[flag], False)
        self.assertEqual(namespace["EXPECTED_COMMIT"], "")
        self.assertEqual(namespace["EXPECTED_SOURCE_SHA256"], "")
        self.assertEqual(namespace["MAX_ADDITIONAL_EPISODES"], 1)
        self.assertEqual(namespace["ALGORITHM"], "qr_dqn_cvar")
        self.assertEqual(namespace["AGENT_SEED"], 11)
        self.assertEqual(namespace["RETURN_SCOPE"], "individual")
        output = PurePosixPath(namespace["OUTPUT_DIRECTORY"])
        self.assertTrue(output.is_absolute())
        self.assertTrue(output.is_relative_to(PurePosixPath("/content/drive/MyDrive")))

    def test_embedded_source_helper_is_exact_repository_source(self):
        tree = ast.parse(code_cells()["verified-source"])
        assignment = next(node for node in tree.body
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "SOURCE_HELPER_CODE"
                                  for target in node.targets))
        embedded = ast.literal_eval(assignment.value)
        expected = (ROOT / "experiments" / "colab_source.py").read_text(encoding="utf-8")
        self.assertEqual(embedded, expected)
        self.assertIn('"__name__": "colab_source_bootstrap"', code_cells()["verified-source"])

    def test_default_gated_cells_never_mount_train_or_run_tests(self):
        cells = code_cells()
        namespace = {}
        exec(compile(cells["configuration"], "configuration", "exec"), namespace)
        runner = Mock(name="run_resumable_development")
        mount = Mock(name="mount")
        runner_module = types.ModuleType("experiments.run_resumable_development")
        runner_module.run_resumable_development = runner
        google_module = types.ModuleType("google")
        colab_module = types.ModuleType("google.colab")
        colab_module.drive = types.SimpleNamespace(mount=mount)
        google_module.colab = colab_module
        modules = {"experiments.run_resumable_development": runner_module,
                   "google": google_module, "google.colab": colab_module}
        with patch.dict(sys.modules, modules), patch("subprocess.run") as subprocess_run, patch("builtins.print"):
            for identifier in ("offline-checks", "drive-opt-in", "development-opt-in"):
                exec(compile(cells[identifier], identifier, "exec"), namespace)
        runner.assert_not_called()
        mount.assert_not_called()
        subprocess_run.assert_not_called()

    def test_development_requires_preflight_and_drive_before_runner_import(self):
        cells = code_cells()
        namespace = {}
        exec(compile(cells["configuration"], "configuration", "exec"), namespace)
        namespace["RUN_DEVELOPMENT"] = True
        runner = Mock(name="run_resumable_development")
        module = types.ModuleType("experiments.run_resumable_development")
        module.run_resumable_development = runner
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                exec(compile(cells["development-opt-in"], "development", "exec"), namespace)
            namespace["PREFLIGHT_READY"] = True
            with self.assertRaisesRegex(RuntimeError, "Drive"):
                exec(compile(cells["development-opt-in"], "development", "exec"), namespace)
        runner.assert_not_called()

    def test_explicit_offline_checks_do_not_target_learning_runners(self):
        cells = code_cells()
        namespace = {}
        exec(compile(cells["configuration"], "configuration", "exec"), namespace)
        namespace.update(RUN_TESTS=True, REPO=ROOT)
        with patch("subprocess.run") as subprocess_run:
            exec(compile(cells["offline-checks"], "checks", "exec"), namespace)
        self.assertEqual(subprocess_run.call_count, 2)
        patterns = []
        for call in subprocess_run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[:6], [sys.executable, "-m", "unittest", "discover", "-s", "tests"])
            self.assertEqual(command[6], "-p")
            patterns.append(command[7])
            self.assertIs(call.kwargs["check"], True)
        self.assertEqual(patterns, ["test_colab*.py", "test_runtime_preflight.py"])

    def test_opted_in_cell_passes_explicit_development_contract(self):
        class ColabPath(PurePosixPath):
            def resolve(self):
                return self

        cells = code_cells()
        namespace = {}
        exec(compile(cells["configuration"], "configuration", "exec"), namespace)
        namespace.update(RUN_DEVELOPMENT=True, PREFLIGHT_READY=True,
                         MOUNT_DRIVE=True, DRIVE_READY=True)
        summary = {"status": "incomplete", "completed_training_episodes": 1,
                   "planned_training_episodes": 4, "completed_validation_episodes": 0,
                   "planned_validation_episodes": 3, "training_slots": 20,
                   "seen_transitions": 10, "gradient_updates": 2,
                   "checkpoint": "steps/000001/agent.pt", "model_sha256": "a" * 64,
                   "main_training_started": False, "test_split_accessed": False}
        full_result = {**summary, "validation": {"large_action_logs": ["retained-in-memory"]}}
        runner = Mock(return_value=full_result)
        module = types.ModuleType("experiments.run_resumable_development")
        module.run_resumable_development = runner
        with patch.dict(sys.modules, {module.__name__: module}), patch("pathlib.Path", ColabPath), \
                patch("builtins.print") as printed:
            exec(compile(cells["development-opt-in"], "development", "exec"), namespace)
        runner.assert_called_once_with(
            output_directory=ColabPath(namespace["OUTPUT_DIRECTORY"]),
            algorithm="qr_dqn_cvar", agent_seed=11, return_scope="individual",
            resume=False, max_additional_episodes=1,
        )
        self.assertIs(namespace["DEVELOPMENT_RESULT"], full_result)
        self.assertEqual(json.loads(printed.call_args.args[0]), summary)


if __name__ == "__main__":
    unittest.main()
