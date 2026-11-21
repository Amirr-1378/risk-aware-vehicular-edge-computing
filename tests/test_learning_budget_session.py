"""Tiny deterministic fixtures for chunk and evaluation budget state transitions."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if HAS_LEARNING:
    from experiments.learning_budget_protocol import LearningBudgetProtocol
    from experiments.learning_runtime import deterministic_cpu
    from experiments.run_learning_budget_session import main, run_learning_budget_session, _episode
    from experiments.run_return_scope_audit import _same_state, _training_state
    from experiments.session_journal import copy_verified_prefix, inspect_session
    from risk_aware_v2x.agents.neural import NeuralCostAgent
    from risk_aware_v2x.scenarios import generate_scenario
    from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol
    from test_return_scope_audit import fixture_plans


def budget_fixture():
    """Six train and three frozen validation episodes per small model chain."""
    previous, original = fixture_plans()
    scenes = replace(original, revision="budget-fixture", split_counts=(6, 2, 2))
    learning = replace(previous.learning, revision="budget-fixture",
        scenario_protocol_sha256=scenes.fingerprint, training_episodes=6,
        validation_profiles=("nominal",), validation_cases=1, agent_seeds=(11,))
    return LearningBudgetProtocol("budget-fixture", learning, budgets=(2, 4, 6),
                                  checkpoint_interval_episodes=3), scenes


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for learning-budget sessions")
class LearningBudgetSessionTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.scenes = budget_fixture()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_session(self, name, **options):
        kwargs = dict(output_directory=self.root / name, protocol=self.plan, scenarios=self.scenes)
        kwargs.update(options)
        return run_learning_budget_session(**kwargs)

    def snapshot(self, name):
        output = self.root / name
        return {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob("*") if p.is_file()}

    def entries(self, name):
        directory = self.root / name
        contract = json.loads((directory / "contract.json").read_text())
        return inspect_session(directory, contract)

    def state(self, name, report):
        encoder = self.plan.encoder_for(report["representation_variant"], max_server_vehicles=2)
        agent, metadata = NeuralCostAgent.load_checkpoint(self.root / name / report["checkpoint"], encoder)
        return _training_state(agent), metadata

    def assert_same_scientific_report(self, a, b):
        # Different pause cadences have different step paths and journal hashes.
        omitted = {"checkpoint", "last_commit_sha256"}
        self.assertEqual({k: v for k, v in a.items() if k not in omitted},
                         {k: v for k, v in b.items() if k not in omitted})

    def test_split_cadences_preserve_model_optimizer_replay_rng_and_evaluation(self):
        for variant in self.plan.variants:
            with self.subTest(variant=variant):
                whole = self.run_session(variant + "-whole", variant=variant, max_additional_episodes=6)
                name = variant + "-split"
                split = self.run_session(name, variant=variant, max_additional_episodes=1)
                self.assertEqual(split["completed_budget_points"], [])
                for _ in range(5):
                    split = self.run_session(name, variant=variant, resume=True, max_additional_episodes=1)
                self.assert_same_scientific_report(whole, split)
                self.assertTrue(_same_state(self.state(variant + "-whole", whole)[0], self.state(name, split)[0]))
                self.assertEqual(whole["seen_transitions"], 36)
                self.assertEqual(whole["gradient_updates"], 35)
                entries = self.entries(variant + "-whole")
                self.assertEqual([e["metadata"]["phase"] for e in entries],
                    ["initialized", "training_chunk", "validation", "training_chunk", "validation", "training_chunk", "validation"])
                self.assertEqual(sum("agent.pt" in e["artifacts"] for e in entries), 4)

    def test_milestone_validation_has_no_effect_on_continuing_training(self):
        report = self.run_session("milestones", max_additional_episodes=6)
        encoder = self.plan.encoder_for("log_snr_v4", max_server_vehicles=2)
        with deterministic_cpu():
            agent = NeuralCostAgent(self.plan.learning.agent_configuration("qr_dqn_cvar"), encoder, seed=11)
            clock = 0
            for index in range(6):
                key = self.scenes.episode_seeds("train", index)
                scene = generate_scenario(self.scenes.profile("nominal"), seed=key.scenario_seed)
                _, clock = _episode(scene, key, encoder, agent, self.plan.learning, clock,
                                    training=True, return_scope="individual")
        self.assertEqual(clock, report["training_slots"])
        self.assertTrue(_same_state(_training_state(agent), self.state("milestones", report)[0]))

    def test_chunk_interval_and_invocation_cap_never_cross_next_milestone(self):
        plan = replace(self.plan, budgets=(4, 6))
        paused = self.run_session("limits", protocol=plan, max_additional_episodes=5)
        self.assertEqual(paused["status"], "incomplete")
        self.assertEqual(paused["completed_training_episodes"], 5)
        self.assertEqual(paused["completed_budget_points"], [4])
        train = [e["metadata"]["completed_training_episodes"] for e in self.entries("limits")
                 if e["metadata"]["phase"] == "training_chunk"]
        self.assertEqual(train, [3, 4, 5])
        final = self.run_session("limits", protocol=plan, resume=True, max_additional_episodes=1)
        self.assertEqual(final["completed_budget_points"], [4, 6])
        self.assertEqual(final["status"], "complete")

    def test_cap_counts_training_only_and_validation_is_frozen(self):
        report = self.run_session("cap", max_additional_episodes=2)
        self.assertEqual(report["completed_training_episodes"], 2)
        self.assertEqual(report["completed_budget_points"], [2])
        self.assertEqual(report["completed_validation_episodes"], 1)
        for episode in report["latest_validation"]["nominal"]:
            self.assertFalse(episode["training"])
            self.assertEqual(episode["updates"], [])
            self.assertEqual(episode["training_slots_before"], report["training_slots"])
            self.assertEqual(episode["training_slots_after"], report["training_slots"])
            self.assertTrue(all(a["epsilon"] == 0 for a in episode["actions"]))
        self.assertFalse(report["main_training_started"])
        self.assertFalse(report["test_split_accessed"])
        self.assertFalse(report["centralized_controller"])

    def test_complete_resume_does_no_episode_or_file_write(self):
        report = self.run_session("complete", max_additional_episodes=6)
        before = self.snapshot("complete")
        with patch("experiments.run_learning_budget_session._episode", side_effect=AssertionError("already complete")):
            self.assertEqual(self.run_session("complete", resume=True), report)
        self.assertEqual(before, self.snapshot("complete"))
        with self.assertRaises(FileExistsError):
            self.run_session("complete")

    def test_contract_source_runtime_or_model_changes_fail_before_work(self):
        self.run_session("bound", max_additional_episodes=1)
        before = self.snapshot("bound")
        changes = [dict(variant="linear_v3"), dict(algorithm="qr_dqn"), dict(agent_seed=29),
                   dict(protocol=replace(self.plan, budgets=(3, 6))),
                   dict(protocol=replace(self.plan, checkpoint_interval_episodes=2))]
        with patch("experiments.run_learning_budget_session._episode", side_effect=AssertionError("too soon")):
            for options in changes:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    self.run_session("bound", resume=True, **options)
        from experiments.runtime_preflight import collect_preflight
        preflight = collect_preflight(require_learning=True)
        for field in ("source", "runtime"):
            changed = json.loads(json.dumps(preflight))
            if field == "source":
                changed["provenance"]["source_sha256"] = "0" * 64
            else:
                changed["runtime_identity"]["numpy_version"] = "0.0.0"
            with patch("experiments.run_learning_budget_session.collect_preflight", return_value=changed):
                with self.assertRaises(ValueError):
                    self.run_session("bound", resume=True)
        self.assertEqual(before, self.snapshot("bound"))

    def test_midchunk_failure_replays_same_seed_prefix_from_last_checkpoint(self):
        calls = []
        def fail_second(*args, **kwargs):
            calls.append(args[1])
            if len(calls) == 2:
                raise RuntimeError("midchunk disconnected")
            return _episode(*args, **kwargs)
        with patch("experiments.run_learning_budget_session._episode", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "midchunk"):
                self.run_session("interrupted", max_additional_episodes=2)
        self.assertEqual(len(self.entries("interrupted")), 1)
        resumed_calls = []
        def track(*args, **kwargs):
            resumed_calls.append((args[1], kwargs["training"]))
            return _episode(*args, **kwargs)
        with patch("experiments.run_learning_budget_session._episode", side_effect=track):
            resumed = self.run_session("interrupted", resume=True, max_additional_episodes=6)
        self.assertEqual(resumed_calls[:2], [(calls[0], True), (calls[1], True)])
        reference = self.run_session("reference", max_additional_episodes=6)
        self.assert_same_scientific_report(resumed, reference)
        self.assertTrue(_same_state(self.state("interrupted", resumed)[0], self.state("reference", reference)[0]))

    def test_failed_checkpoint_requires_verified_prefix_recovery_to_new_directory(self):
        self.run_session("broken", max_additional_episodes=1)
        before = self.snapshot("broken")
        def interrupted_save(agent, path, **kwargs):
            Path(path).write_bytes(b"partial checkpoint")
            raise RuntimeError("checkpoint interrupted")
        with patch.object(NeuralCostAgent, "save_checkpoint", interrupted_save):
            with self.assertRaisesRegex(RuntimeError, "checkpoint interrupted"):
                self.run_session("broken", resume=True, max_additional_episodes=1)
        evidence = self.snapshot("broken")
        self.assertTrue(all(evidence[k] == v for k, v in before.items()))
        with self.assertRaises(ValueError):
            self.run_session("broken", resume=True)
        contract = json.loads((self.root / "broken" / "contract.json").read_text())
        self.assertEqual(copy_verified_prefix(self.root / "broken", self.root / "recovered", contract), 2)
        resumed = self.run_session("recovered", resume=True, max_additional_episodes=5)
        reference = self.run_session("reference", max_additional_episodes=6)
        self.assert_same_scientific_report(resumed, reference)
        self.assertTrue(_same_state(self.state("recovered", resumed)[0], self.state("reference", reference)[0]))
        self.assertEqual(evidence, self.snapshot("broken"))

    def test_validation_failure_retries_due_cohort_before_next_training(self):
        def fail_validation(*args, **kwargs):
            if not kwargs["training"]:
                raise RuntimeError("validation interrupted")
            return _episode(*args, **kwargs)
        with patch("experiments.run_learning_budget_session._episode", side_effect=fail_validation):
            with self.assertRaisesRegex(RuntimeError, "validation interrupted"):
                self.run_session("validation", max_additional_episodes=2)
        self.assertEqual([e["metadata"]["phase"] for e in self.entries("validation")], ["initialized", "training_chunk"])
        calls = []
        def track(*args, **kwargs):
            calls.append(kwargs["training"])
            return _episode(*args, **kwargs)
        with patch("experiments.run_learning_budget_session._episode", side_effect=track):
            report = self.run_session("validation", resume=True, max_additional_episodes=2)
        self.assertEqual(calls, [False, True, True, False])
        self.assertEqual(report["completed_budget_points"], [2, 4])
        self.assertEqual(report["completed_training_episodes"], 4)

    def test_validation_mutation_fails_before_publishing_cohort(self):
        def mutate(*args, **kwargs):
            report, clock = _episode(*args, **kwargs)
            if not kwargs["training"]:
                args[3].gradient_updates += 1
            return report, clock
        with patch("experiments.run_learning_budget_session._episode", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "mutated"):
                self.run_session("mutated", max_additional_episodes=2)
        self.assertEqual([e["metadata"]["phase"] for e in self.entries("mutated")], ["initialized", "training_chunk"])

    def test_only_train_validation_seed_keys_are_used(self):
        original, accesses = DevelopmentProtocol.episode_seeds, []
        def guard(instance, split, index):
            self.assertNotEqual(split, "test")
            accesses.append(split)
            return original(instance, split, index)
        with patch.object(DevelopmentProtocol, "episode_seeds", guard):
            self.run_session("seeds", max_additional_episodes=2)
        self.assertEqual(set(accesses), {"train", "validation"})

    def test_plan_only_creates_no_learner_trajectory_or_output(self):
        output = self.root / "absent"
        with patch.object(NeuralCostAgent, "__init__", side_effect=AssertionError("no learner")), \
                patch("experiments.run_learning_budget_session.generate_scenario", side_effect=AssertionError("no trajectory")), \
                contextlib.redirect_stdout(io.StringIO()) as stdout:
            main(["--plan", "--output-dir", str(output)])
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["planning_only"])
        self.assertEqual(report["total_physical_episode_executions"], 9280)
        self.assertFalse(output.exists())

    def test_invalid_invocations_create_no_output(self):
        for options in (dict(max_additional_episodes=0), dict(max_additional_episodes=True),
                        dict(max_additional_episodes=1.5), dict(resume=1), dict(agent_seed=True)):
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError)):
                self.run_session("absent", **options)
            self.assertFalse((self.root / "absent").exists())


if __name__ == "__main__":
    unittest.main()
