"""Episode persistence must reproduce the unchanged single-model learner."""

import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if HAS_LEARNING:
    from experiments.run_resumable_development import run_resumable_development
    from experiments.run_return_scope_audit import _same_state, _training_state, run_return_scope_audit
    from experiments.session_journal import copy_verified_prefix, inspect_session
    from risk_aware_v2x.agents import ObservationEncoder
    from risk_aware_v2x.agents.neural import NeuralCostAgent
    from risk_aware_v2x.agents.team_neural import TeamNeuralCostAgent
    from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol
    from test_return_scope_audit import fixture_plans


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for session audit")
class ResumableDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.paired, self.scenarios = fixture_plans()
        self.learning = replace(self.paired.learning, algorithms=("qr_dqn_cvar",), agent_seeds=(11,),
                                validation_profiles=("nominal",), validation_cases=1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_session(self, name, **kwargs):
        options = dict(output_directory=self.root / name, learning=self.learning, scenarios=self.scenarios)
        options.update(kwargs)
        return run_resumable_development(**options)

    def state(self, name, report, scope):
        cls = NeuralCostAgent if scope == "individual" else TeamNeuralCostAgent
        encoder = ObservationEncoder(max_server_vehicles=2, scales=self.learning.feature_scales)
        agent, metadata = cls.load_checkpoint(self.root / name / report["checkpoint"], encoder)
        return _training_state(agent), metadata

    def snapshot(self, name):
        output = self.root / name
        return {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob("*") if p.is_file()}

    def assert_equivalent_reports(self, first, second):
        # Torch storage sharing can serialize differently after load. Artifact
        # hashes audit each file; exact scientific state is compared separately.
        self.assertEqual({k: v for k, v in first.items() if k != "last_commit_sha256"},
                         {k: v for k, v in second.items() if k != "last_commit_sha256"})

    def test_split_resume_matches_uninterrupted_all_state_and_original_runtime(self):
        original = run_return_scope_audit(protocol=replace(self.paired, learning=self.learning), scenarios=self.scenarios)
        for scope in ("individual", "team"):
            with self.subTest(scope=scope):
                uninterrupted = self.run_session(scope + "-whole", return_scope=scope, max_additional_episodes=2)
                paused = self.run_session(scope + "-split", return_scope=scope)
                self.assertEqual(paused["status"], "incomplete")
                self.assertIsNone(paused["validation"])
                self.assertEqual(paused["completed_validation_episodes"], 0)
                self.assertEqual(paused["completed_training_episodes"], 1)
                continued = self.run_session(scope + "-split", return_scope=scope, resume=True)
                self.assert_equivalent_reports(continued, uninterrupted)
                a, _ = self.state(scope + "-whole", uninterrupted, scope)
                b, _ = self.state(scope + "-split", continued, scope)
                self.assertTrue(_same_state(a, b))
                existing = next(run for run in original["runs"] if run["return_scope"] == scope)
                for key in ("model_sha256", "initial_model_sha256", "seen_transitions", "gradient_updates", "validation"):
                    self.assertEqual(continued[key], json.loads(json.dumps(existing[key])))
                contract = json.loads((self.root / (scope + "-whole") / "contract.json").read_text())
                entries = inspect_session(self.root / (scope + "-whole"), contract)
                self.assertEqual([e["metadata"]["phase"] for e in entries],
                                 ["initialized", "training", "training", "validation"])

    def test_complete_resume_does_not_run_episodes_or_write_files(self):
        report = self.run_session("complete", max_additional_episodes=2)
        before = self.snapshot("complete")
        with patch("experiments.run_resumable_development._episode", side_effect=AssertionError("already complete")):
            self.assertEqual(self.run_session("complete", resume=True), report)
        self.assertEqual(before, self.snapshot("complete"))
        with self.assertRaises(FileExistsError):
            self.run_session("complete")

    def test_contract_changes_fail_before_writing_or_running(self):
        self.run_session("bound")
        before = self.snapshot("bound")
        changes = [dict(return_scope="team"), dict(algorithm="qr_dqn"), dict(agent_seed=29),
                   dict(learning=replace(self.learning, training_episodes=1)),
                   dict(learning=replace(self.learning, common_agent=replace(self.learning.common_agent, learning_rate=.002)))]
        for changeset in changes:
            with self.subTest(changes=changeset), self.assertRaises(ValueError):
                self.run_session("bound", resume=True, **changeset)
        from experiments.runtime_preflight import collect_preflight
        real_preflight = collect_preflight(require_learning=True)
        for field in ("source", "runtime"):
            changed = json.loads(json.dumps(real_preflight))
            if field == "source":
                changed["provenance"]["source_sha256"] = "0" * 64
            else:
                changed["runtime_identity"]["numpy_version"] = "2.0.0"
            with patch("experiments.run_resumable_development.collect_preflight", return_value=changed):
                with self.assertRaises(ValueError):
                    self.run_session("bound", resume=True)
        self.assertEqual(before, self.snapshot("bound"))

    def test_interruption_with_no_generation_reruns_same_scene_seed(self):
        original = self.run_session("reference", max_additional_episodes=2)
        with patch("experiments.run_resumable_development._episode", side_effect=RuntimeError("runtime disconnected")):
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                self.run_session("interrupted")
        continued = self.run_session("interrupted", resume=True, max_additional_episodes=2)
        self.assert_equivalent_reports(original, continued)

    def test_failed_checkpoint_leaves_evidence_and_explicit_prefix_recovery(self):
        self.run_session("broken")
        before = self.snapshot("broken")

        def interrupted_save(agent, path, **kwargs):
            Path(path).write_bytes(b"partial checkpoint")
            raise RuntimeError("checkpoint interrupted")

        with patch.object(NeuralCostAgent, "save_checkpoint", interrupted_save):
            with self.assertRaisesRegex(RuntimeError, "checkpoint interrupted"):
                self.run_session("broken", resume=True)
        evidence = self.snapshot("broken")
        self.assertTrue(all(evidence[k] == v for k, v in before.items()))
        with self.assertRaises(ValueError):
            self.run_session("broken", resume=True)
        contract = json.loads((self.root / "broken" / "contract.json").read_text())
        self.assertEqual(copy_verified_prefix(self.root / "broken", self.root / "recovered", contract), 2)
        recovered = self.run_session("recovered", resume=True)
        reference = self.run_session("recovery-reference", max_additional_episodes=2)
        self.assert_equivalent_reports(recovered, reference)
        self.assertEqual(evidence, self.snapshot("broken"))

    def test_failed_validation_retries_same_final_frozen_model(self):
        self.run_session("validation")
        from experiments.run_resumable_development import _episode

        def fail_validation(*args, **kwargs):
            if not kwargs["training"]:
                raise RuntimeError("validation interrupted")
            return _episode(*args, **kwargs)

        with patch("experiments.run_resumable_development._episode", side_effect=fail_validation):
            with self.assertRaisesRegex(RuntimeError, "validation interrupted"):
                self.run_session("validation", resume=True)
        reference = self.run_session("validation-reference", max_additional_episodes=2)
        resumed = self.run_session("validation", resume=True)
        self.assert_equivalent_reports(resumed, reference)

    def test_checkpoint_metadata_and_state_counters_must_agree(self):
        self.run_session("mismatch")
        before = self.snapshot("mismatch")
        original_load = NeuralCostAgent.load_checkpoint

        def wrong_state(*args, **kwargs):
            agent, metadata = original_load(*args, **kwargs)
            agent.seen_transitions += 1
            return agent, metadata

        with patch.object(NeuralCostAgent, "load_checkpoint", side_effect=wrong_state):
            with self.assertRaisesRegex(ValueError, "counter mismatch"):
                self.run_session("mismatch", resume=True)
        self.assertEqual(before, self.snapshot("mismatch"))

    def test_train_validation_only_and_frozen_evaluation(self):
        original, accesses = DevelopmentProtocol.episode_seeds, []

        def guard(instance, split, index):
            self.assertNotEqual(split, "test")
            accesses.append(split)
            return original(instance, split, index)

        with patch.object(DevelopmentProtocol, "episode_seeds", guard):
            report = self.run_session("boundary", max_additional_episodes=2)
        self.assertEqual(set(accesses), {"train", "validation"})
        self.assertFalse(report["main_training_started"])
        self.assertFalse(report["test_split_accessed"])
        self.assertFalse(report["centralized_controller"])
        self.assertEqual(report["training_behavior"], "standard-learned-epsilon")
        for episodes in report["validation"].values():
            self.assertTrue(all(not episode["updates"] for episode in episodes))

    def test_invalid_invocation_fails_without_output(self):
        for options in (dict(max_additional_episodes=0), dict(max_additional_episodes=True),
                        dict(return_scope="centralized"), dict(resume=1)):
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError)):
                self.run_session("absent", **options)
            self.assertFalse((self.root / "absent").exists())

    def test_persisted_episode_rejects_wrong_phase_seed_and_teacher_action(self):
        self.run_session("report")
        from experiments.run_resumable_development import _check_episode
        from risk_aware_v2x.scenarios import generate_scenario
        report = json.loads((self.root / "report" / "steps" / "000001" / "report.json").read_text())["episode"]
        key = self.scenarios.episode_seeds("train", 0)
        scene = generate_scenario(self.scenarios.profile(self.learning.training_profile), seed=key.scenario_seed)
        for changes in (dict(training=False), dict(training=1), dict(environment_seed=999),
                        dict(training_slots_before=False)):
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                _check_episode({**report, **changes}, scene, key, clock=0, training=True)
        report["actions"][0]["behavior_source"] = "cached_combined_cost_greedy"
        with self.assertRaises(ValueError):
            _check_episode(report, scene, key, clock=0, training=True)


if __name__ == "__main__":
    unittest.main()
