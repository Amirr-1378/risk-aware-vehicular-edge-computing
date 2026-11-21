"""Tiny complete-grid curves, immutable evidence and read-only collection."""

import hashlib
import importlib.util
import json
import random
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if HAS_LEARNING:
    import torch
    from experiments.learning_budget_protocol import FIXED_POLICY_NAMES, LearningBudgetProtocol
    from experiments.learning_budget_results import run_learning_budget_references, summarize_learning_budget
    from experiments.run_learning_budget_session import run_learning_budget_session
    from experiments.session_journal import inspect_session
    from test_return_scope_audit import fixture_plans


def result_fixture():
    scope, scenes = fixture_plans()
    learning = replace(scope.learning, agent_seeds=(11,), validation_cases=1,
                       validation_profiles=("nominal",))
    return LearningBudgetProtocol("budget-result-fixture", learning, budgets=(1, 2),
                                  checkpoint_interval_episodes=2), scenes


def _snapshot(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for learning-budget results")
class LearningBudgetResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.study = cls.root / "complete"
        cls.plan, cls.scenes = result_fixture()
        for variant in cls.plan.variants:
            for algorithm in cls.plan.learning.algorithms:
                run_learning_budget_session(output_directory=cls.chain(cls.study, variant, algorithm),
                    protocol=cls.plan, scenarios=cls.scenes, variant=variant,
                    algorithm=algorithm, agent_seed=11, max_additional_episodes=2)
        cls.reference = run_learning_budget_references(output_directory=cls.study / "references",
                                                       protocol=cls.plan, scenarios=cls.scenes)

    @staticmethod
    def chain(study, variant="linear_v3", algorithm="qr_dqn"):
        return study / variant / algorithm / "seed-11"

    def collect(self, directory=None, **kwargs):
        return summarize_learning_budget(study_directory=directory or self.study,
            protocol=self.plan, scenarios=self.scenes, **kwargs)

    def copy_study(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "study"
        shutil.copytree(self.study, destination)
        return destination

    def rewrite_final_validation(self, study, operation, *, metadata=False):
        session = self.chain(study)
        contract = json.loads((session / "contract.json").read_text())
        entries = inspect_session(session, contract)
        step = session / "steps" / f'{entries[-1]["step_index"]:06d}'
        report_path, commit_path = step / "report.json", step / "commit.json"
        report, commit = json.loads(report_path.read_text()), json.loads(commit_path.read_text())
        operation(commit["metadata"] if metadata else report)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        commit["artifacts"]["report.json"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
        commit_path.write_text(json.dumps(commit), encoding="utf-8")

    def test_complete_grid_keeps_each_model_budget_and_profile_separate(self):
        report = self.collect()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["progress"]), 4)
        self.assertTrue(all(row["completed_budget_points"] == [1, 2] for row in report["progress"]))
        self.assertEqual(len(report["model_curves"]), 8)
        self.assertEqual({row["budget"] for row in report["model_curves"]}, {1, 2})
        self.assertEqual(set(report["representation_comparisons"]), {"1", "2"})
        for comparison in report["representation_comparisons"].values():
            self.assertEqual(len(comparison["model_cost_risks"]), 4)
            self.assertEqual(len(comparison["paired_variant_differences"]), 2)
            self.assertTrue(comparison["validation_scene_pairing_verified"])
            self.assertTrue(all(row["episode_cost_risk"]["episodes"] == 1
                                for row in comparison["model_cost_risks"]))
        self.assertEqual(set(report["fixed_references"]["nominal"]), set(FIXED_POLICY_NAMES))
        self.assertEqual(report["selection_rule"], "fixed-final-budget-no-validation-selection")
        self.assertFalse(report["main_training_started"])
        self.assertFalse(report["test_split_accessed"])

    def test_missing_or_partial_chains_have_progress_only(self):
        study = self.copy_study()
        shutil.rmtree(self.chain(study))
        report = self.collect(study)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(sum(row["status"] == "missing" for row in report["progress"]), 1)
        for name in ("model_curves", "representation_comparisons", "fixed_references"):
            self.assertIsNone(report[name])
        run_learning_budget_session(output_directory=self.chain(study), protocol=self.plan,
            scenarios=self.scenes, variant="linear_v3", algorithm="qr_dqn",
            agent_seed=11, max_additional_episodes=1)
        report = self.collect(study)
        self.assertEqual(report["status"], "incomplete")
        partial = next(row for row in report["progress"] if row["status"] == "incomplete")
        self.assertEqual(partial["completed_budget_points"], [1])
        self.assertIsNone(report["model_curves"])

    def test_missing_fixed_references_are_explicit_after_complete_learned_grid(self):
        study = self.copy_study()
        shutil.rmtree(study / "references")
        report = self.collect(study)
        self.assertEqual(report["status"], "awaiting_references")
        self.assertEqual(len(report["model_curves"]), 8)
        self.assertIsNone(report["fixed_references"])

    def test_reference_suite_runs_once_on_the_exact_declared_cohort(self):
        reference = self.reference
        self.assertEqual(reference["physical_episode_executions"], 5)
        self.assertEqual(reference["learning_budget_protocol_sha256"], self.plan.fingerprint)
        self.assertEqual(reference["scenario_protocol_sha256"], self.scenes.fingerprint)
        self.assertFalse(reference["training_executed"])
        self.assertFalse(reference["test_split_accessed"])
        self.assertFalse(reference["main_training_started"])
        audit = reference["references"]["nominal"]
        self.assertEqual(set(audit["policies"]), set(FIXED_POLICY_NAMES))
        self.assertEqual(audit["scenarios"][0]["seeds"]["split"], "validation")
        self.assertTrue(all(len(episodes) == 1 for episodes in audit["policies"].values()))
        with self.assertRaises(FileExistsError):
            run_learning_budget_references(output_directory=self.study / "references",
                                           protocol=self.plan, scenarios=self.scenes)

    def test_reference_identity_and_scene_mismatches_fail(self):
        changes = (
            lambda value: value.update(source_sha256="0" * 64),
            lambda value: value.update(purpose="unrelated-report"),
            lambda value: value.update(model_contract={}),
            lambda value: value.update(main_training_started=True),
            lambda value: value["references"]["nominal"]["scenarios"][0].update(fingerprint="changed-scene"),
            lambda value: value["references"]["nominal"]["policies"].pop("always_mec"),
        )
        for change in changes:
            with self.subTest(change=change):
                study = self.copy_study()
                path = study / "references" / "report.json"
                value = json.loads(path.read_text())
                change(value)
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.collect(study)

    def test_reference_identity_change_during_run_cannot_publish_a_report(self):
        provenance, identity = self.reference["provenance"], self.reference["runtime_identity"]
        cases = (
            ("source", [provenance, {**provenance, "source_sha256": "0" * 64}], [identity, identity]),
            ("runtime", [provenance, provenance], [identity, {**identity, "torch_version": "changed"}]),
        )
        for name, source_values, runtime_values in cases:
            with self.subTest(changed=name), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "references"
                with patch("experiments.learning_budget_results.source_provenance", side_effect=source_values), \
                     patch("experiments.learning_budget_results.runtime_identity", side_effect=runtime_values), \
                     patch("experiments.learning_budget_results.run_scenario_audit",
                           return_value=self.reference["references"]["nominal"]) as audit:
                    with self.assertRaisesRegex(RuntimeError, "source or runtime changed"):
                        run_learning_budget_references(output_directory=output,
                                                       protocol=self.plan, scenarios=self.scenes)
                self.assertEqual(audit.call_count, 1)
                self.assertFalse((output / "report.json").exists())
                self.assertEqual(list(output.iterdir()), [])

    def test_changed_validation_fingerprint_fails_even_with_rehashed_artifact(self):
        study = self.copy_study()
        self.rewrite_final_validation(study,
            lambda value: value["validation"]["nominal"][0].update(scenario_fingerprint="different-scene"))
        with self.assertRaisesRegex(ValueError, "cohort|unpaired"):
            self.collect(study)

    def test_validation_cannot_update_train_or_change_clock(self):
        changes = (
            lambda value: value["validation"]["nominal"][0].update(training=True),
            lambda value: value["validation"]["nominal"][0].update(updates=[{}]),
            lambda value: value["validation"]["nominal"][0].update(training_slots_after=99999),
            lambda value: value["validation"]["nominal"][0]["actions"][0].update(epsilon=.1),
            lambda value: value["validation"]["nominal"][0]["seeds"].update(scenario_seed=0),
        )
        for change in changes:
            with self.subTest(change=change):
                study = self.copy_study()
                self.rewrite_final_validation(study, change)
                with self.assertRaises(ValueError):
                    self.collect(study)

    def test_collector_rejects_corrupt_checkpoint_before_summarizing(self):
        study = self.copy_study()
        checkpoint = next(self.chain(study).glob("steps/*/agent.pt"))
        checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.collect(study)

    def test_cross_chain_source_and_runtime_must_match_before_comparison(self):
        for field, changed in (("source_sha256", "0" * 64), ("runtime_identity", {})):
            with self.subTest(field=field):
                study = self.copy_study()
                path = self.chain(study, algorithm="qr_dqn_cvar") / "contract.json"
                contract = json.loads(path.read_text())
                contract[field] = changed
                path.write_text(json.dumps(contract), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "source/runtime"):
                    self.collect(study)

    def test_declared_chain_cannot_be_substituted_with_another_seed_or_encoder(self):
        changes = (lambda value: value.update(agent_seed=29),
                   lambda value: value["encoder"].update(schema_version=99))
        for change in changes:
            with self.subTest(change=change):
                study = self.copy_study()
                path = self.chain(study) / "contract.json"
                contract = json.loads(path.read_text())
                change(contract)
                path.write_text(json.dumps(contract), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    self.collect(study)

    def test_collector_is_read_only_and_never_loads_or_runs_a_model(self):
        before = _snapshot(self.study)
        python_rng, torch_rng = random.getstate(), torch.random.get_rng_state().clone()
        failure = AssertionError("collector must only read existing evidence")
        with patch("risk_aware_v2x.agents.neural.NeuralCostAgent.__init__", side_effect=failure), \
             patch("risk_aware_v2x.agents.neural.NeuralCostAgent.load_checkpoint", side_effect=failure), \
             patch("experiments.run_learning_budget_session.generate_scenario", side_effect=failure), \
             patch("risk_aware_v2x.environment.simulator.SlottedEnvironment.step", side_effect=failure), \
             patch("experiments.learning_budget_results.run_scenario_audit", side_effect=failure):
            report = self.collect()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(before, _snapshot(self.study))
        self.assertEqual(python_rng, random.getstate())
        self.assertTrue(torch.equal(torch_rng, torch.random.get_rng_state()))

    def test_summary_output_is_exclusive_and_cannot_modify_input_sessions(self):
        study = self.copy_study()
        output = study / "summary.json"
        report = self.collect(study, output_path=output)
        self.assertEqual(json.loads(output.read_text()), report)
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.collect(study, output_path=output)
        self.assertEqual(before, output.read_bytes())
        for path in (self.chain(study) / "summary.json", study / "references" / "summary.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.collect(study, output_path=path)
            self.assertFalse(path.exists())

    def test_invalid_study_binding_fails_before_fixed_scenarios_or_output(self):
        bad = replace(self.plan, learning=replace(self.plan.learning, scenario_protocol_sha256="0" * 64))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "absent"
            with patch("experiments.learning_budget_results.run_scenario_audit", side_effect=AssertionError("too soon")):
                with self.assertRaises(ValueError):
                    run_learning_budget_references(output_directory=output, protocol=bad, scenarios=self.scenes)
            self.assertFalse(output.exists())
