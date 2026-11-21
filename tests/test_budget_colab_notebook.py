"""Offline notebook preparation checks; never mount Drive or execute learning."""

import ast
import builtins
import json
import posixpath
import sys
import types
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_learning_budget.ipynb"


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def cells():
    return {cell["id"]: "".join(cell["source"])
            for cell in notebook()["cells"] if cell["cell_type"] == "code"}


def configuration():
    namespace = {}
    exec(compile(cells()["configuration"], "configuration", "exec"), namespace)
    return namespace


class ColabPath(PurePosixPath):
    def resolve(self):
        return type(self)(posixpath.normpath(str(self)))

    def is_dir(self):
        return self == type(self)("/content/drive/MyDrive")


class BudgetColabNotebookTests(unittest.TestCase):
    def test_nbformat_cells_compile_and_have_no_execution_or_outputs(self):
        record = notebook()
        self.assertEqual(record["nbformat"], 4)
        self.assertGreaterEqual(record["nbformat_minor"], 5)
        self.assertEqual(record["metadata"]["kernelspec"]["name"], "python3")
        ids = [cell["id"] for cell in record["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for identifier, source in cells().items():
            with self.subTest(cell=identifier):
                compile(source, f"{NOTEBOOK.name}:{identifier}", "exec")
        for cell in record["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_placeholders_and_manual_single_chain_defaults(self):
        values = configuration()
        for flag in ("RUN_TESTS", "MOUNT_DRIVE", "RUN_STUDY", "RESUME",
                     "PREFLIGHT_READY", "PLAN_READY", "DRIVE_READY"):
            self.assertIs(values[flag], False)
        self.assertEqual(values["EXPECTED_COMMIT"], "")
        self.assertEqual(values["EXPECTED_SOURCE_SHA256"], "")
        self.assertEqual(values["ARCHIVE_PATH"], "/content/source.zip")
        self.assertEqual(values["SOURCE_DESTINATION"], "/content/risk-aware-v2x-budget-source")
        self.assertEqual(values["VARIANT"], "log_snr_v4")
        self.assertEqual(values["ALGORITHM"], "qr_dqn_cvar")
        self.assertEqual(values["AGENT_SEED"], 11)
        self.assertEqual(values["MAX_ADDITIONAL_EPISODES"], 10)
        output = PurePosixPath(values["STUDY_OUTPUT_ROOT"])
        self.assertTrue(output.is_absolute())
        self.assertTrue(output.is_relative_to(PurePosixPath("/content/drive/MyDrive")))

    def test_bootstrap_embeds_exact_current_repository_verifier(self):
        source = cells()["verified-source"]
        assignment = next(node for node in ast.parse(source).body
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "SOURCE_HELPER_CODE"
                                  for target in node.targets))
        self.assertEqual(ast.literal_eval(assignment.value),
                         (ROOT / "experiments" / "colab_source.py").read_text(encoding="utf-8"))
        self.assertIn('"__name__": "colab_source_bootstrap"', source)

    def test_default_action_cells_do_not_import_driver_mount_or_run_tests(self):
        namespace = configuration()
        original = builtins.__import__
        forbidden = {"experiments.run_learning_budget_session", "google.colab"}

        def guarded_import(name, *args, **kwargs):
            if name in forbidden:
                raise AssertionError("default action gates must not import execution helpers")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import), \
                patch("subprocess.run") as subprocess_run, patch("builtins.print"):
            for identifier in ("offline-checks", "drive-opt-in", "study-opt-in"):
                exec(compile(cells()[identifier], identifier, "exec"), namespace)
        subprocess_run.assert_not_called()

    def test_planner_requires_preflight_before_importing_protocols(self):
        namespace = configuration()
        original = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "experiments.learning_budget_protocol":
                raise AssertionError("planner imports only after successful preflight")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                exec(compile(cells()["study-plan"], "plan", "exec"), namespace)

    def test_planner_only_loads_validates_and_describes_declared_metadata(self):
        namespace = configuration()
        namespace["PREFLIGHT_READY"] = True
        scenario = object()
        plan = Mock()
        metadata = {"model_chains": 12, "training_episode_executions": 6000,
                    "learned_validation_episode_executions": 2880,
                    "fixed_policy_episode_executions": 400,
                    "total_physical_episode_executions": 9280}
        plan.execution_plan.return_value = metadata
        plan_class = types.SimpleNamespace(load=Mock(return_value=plan))
        scenario_class = types.SimpleNamespace(load=Mock(return_value=scenario))
        protocol_module = types.ModuleType("experiments.learning_budget_protocol")
        protocol_module.LearningBudgetProtocol = plan_class
        protocol_module.DEFAULT_LEARNING_BUDGET_PROTOCOL = "learning-budget.json"
        protocol_module.DEFAULT_BUDGET_SCENARIO_PROTOCOL = "scenario-budget.json"
        scenario_module = types.ModuleType("risk_aware_v2x.scenarios.protocol")
        scenario_module.DevelopmentProtocol = scenario_class
        modules = {protocol_module.__name__: protocol_module, scenario_module.__name__: scenario_module}
        with patch.dict(sys.modules, modules), patch("builtins.print") as printed:
            exec(compile(cells()["study-plan"], "plan", "exec"), namespace)
        plan_class.load.assert_called_once_with("learning-budget.json")
        scenario_class.load.assert_called_once_with("scenario-budget.json")
        plan.validate_against.assert_called_once_with(scenario)
        plan.execution_plan.assert_called_once_with()
        self.assertIs(namespace["BUDGET_PROTOCOL"], plan)
        self.assertIs(namespace["BUDGET_SCENARIOS"], scenario)
        self.assertIs(namespace["PLAN_READY"], True)
        self.assertEqual(json.loads(printed.call_args.args[0]), metadata)
        self.assertNotIn("STUDY_RESULT", namespace)

    def test_setup_checks_use_only_three_nonlearning_unittest_patterns(self):
        namespace = configuration()
        namespace.update(RUN_TESTS=True, REPO=ROOT)
        with patch("subprocess.run") as subprocess_run:
            exec(compile(cells()["offline-checks"], "checks", "exec"), namespace)
        self.assertEqual(subprocess_run.call_count, 3)
        patterns = []
        for call in subprocess_run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[:6], [sys.executable, "-m", "unittest", "discover", "-s", "tests"])
            self.assertEqual(command[6], "-p")
            patterns.append(command[7])
            self.assertIs(call.kwargs["check"], True)
            self.assertEqual(call.kwargs["cwd"], ROOT)
        self.assertEqual(patterns, ["test_budget_colab_notebook.py", "test_colab_source.py", "test_runtime_preflight.py"])

    def test_study_requires_preflight_plan_and_drive_before_runner_import(self):
        namespace = configuration()
        namespace["RUN_STUDY"] = True
        runner = Mock()
        module = types.ModuleType("experiments.run_learning_budget_session")
        module.run_learning_budget_session = runner
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(RuntimeError, "preflight"):
                exec(compile(cells()["study-opt-in"], "run", "exec"), namespace)
            namespace["PREFLIGHT_READY"] = True
            with self.assertRaisesRegex(RuntimeError, "plan"):
                exec(compile(cells()["study-opt-in"], "run", "exec"), namespace)
            namespace["PLAN_READY"] = True
            with self.assertRaisesRegex(RuntimeError, "Drive"):
                exec(compile(cells()["study-opt-in"], "run", "exec"), namespace)
        runner.assert_not_called()

    def test_opted_in_call_runs_one_chain_with_explicit_fixed_protocol(self):
        namespace = configuration()
        plan = types.SimpleNamespace(variants=("linear_v3", "log_snr_v4"),
            learning=types.SimpleNamespace(algorithms=("qr_dqn", "qr_dqn_cvar"), agent_seeds=(11, 29, 47)))
        scenario = object()
        namespace.update(RUN_STUDY=True, PREFLIGHT_READY=True, PLAN_READY=True,
                         MOUNT_DRIVE=True, DRIVE_READY=True, BUDGET_PROTOCOL=plan,
                         BUDGET_SCENARIOS=scenario)
        summary = {"status": "incomplete", "completed_training_episodes": 10,
            "planned_training_episodes": 500, "completed_budget_points": [],
            "planned_budget_points": [20, 100, 500], "latest_validation_budget": None,
            "completed_validation_episodes": 0, "planned_validation_episodes": 240,
            "training_slots": 2100, "seen_transitions": 175, "gradient_updates": 144,
            "checkpoint": "steps/000001/agent.pt", "model_sha256": "a" * 64,
            "main_training_started": False, "test_split_accessed": False}
        result = {**summary, "latest_validation": {"full_logs": "kept-in-memory"}}
        runner = Mock(return_value=result)
        module = types.ModuleType("experiments.run_learning_budget_session")
        module.run_learning_budget_session = runner
        with patch.dict(sys.modules, {module.__name__: module}), patch("pathlib.Path", ColabPath), \
                patch("builtins.print") as printed:
            exec(compile(cells()["study-opt-in"], "run", "exec"), namespace)
        runner.assert_called_once_with(
            output_directory=ColabPath(namespace["STUDY_OUTPUT_ROOT"]) / "log_snr_v4" / "qr_dqn_cvar" / "seed-11",
            protocol=plan, scenarios=scenario, variant="log_snr_v4", algorithm="qr_dqn_cvar", agent_seed=11,
            resume=False, max_additional_episodes=10,
        )
        self.assertIs(namespace["STUDY_RESULT"], result)
        self.assertEqual(json.loads(printed.call_args.args[0]), summary)
        self.assertFalse(any(isinstance(node, (ast.For, ast.While))
                             for node in ast.walk(ast.parse(cells()["study-opt-in"]))))

    def test_invalid_selection_or_output_path_is_rejected_without_calling_runner(self):
        plan = types.SimpleNamespace(variants=("linear_v3", "log_snr_v4"),
            learning=types.SimpleNamespace(algorithms=("qr_dqn", "qr_dqn_cvar"), agent_seeds=(11, 29, 47)))
        changes = [dict(VARIANT="../../outside"), dict(ALGORITHM="unknown"), dict(AGENT_SEED=True),
                   dict(AGENT_SEED=99), dict(STUDY_OUTPUT_ROOT="relative/output"),
                   dict(STUDY_OUTPUT_ROOT="/tmp/output"), dict(STUDY_OUTPUT_ROOT="/content/drive/MyDrive"),
                   dict(STUDY_OUTPUT_ROOT="/content/drive/MyDrive/../../outside")]
        runner = Mock()
        module = types.ModuleType("experiments.run_learning_budget_session")
        module.run_learning_budget_session = runner
        with patch.dict(sys.modules, {module.__name__: module}), patch("pathlib.Path", ColabPath):
            for change in changes:
                namespace = configuration()
                namespace.update(RUN_STUDY=True, PREFLIGHT_READY=True, PLAN_READY=True,
                                 MOUNT_DRIVE=True, DRIVE_READY=True, BUDGET_PROTOCOL=plan,
                                 BUDGET_SCENARIOS=object(), **change)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    exec(compile(cells()["study-opt-in"], "run", "exec"), namespace)
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
