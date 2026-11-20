"""Paired training-behavior variant preserves core methods and fixed total budgets."""

import copy
import importlib.util
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

HAS_LEARNING = all(importlib.util.find_spec(x) is not None for x in ("torch", "numpy"))
if HAS_LEARNING:
    import torch
    from experiments.behavior_warmup_protocol import BehaviorWarmupProtocol
    from experiments.run_behavior_warmup_audit import load_behavior_checkpoint, run_behavior_warmup_audit
    from experiments.run_return_scope_audit import run_return_scope_audit
    from risk_aware_v2x.agents import ObservationEncoder
    from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol
    from test_return_scope_audit import fixture_plans


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for behavior warmup audit")
class BehaviorWarmupAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paired, cls.scenarios = fixture_plans()
        cls.protocol = BehaviorWarmupProtocol("behavior-fixture", paired, teacher_episodes=1)
        cls.report = run_behavior_warmup_audit(protocol=cls.protocol, scenarios=cls.scenarios)

    def test_same_total_budget_matching_initial_weights_and_strict_teacher_boundary(self):
        self.assertEqual(len(self.report["runs"]), 16)
        self.assertEqual(self.report["physical_episode_executions"], 116)
        self.assertEqual(self.report["training_budget"]["extra_teacher_episodes"], 0)
        initials = {}
        for run in self.report["runs"]:
            pair = run["algorithm"], run["agent_seed"]
            self.assertEqual(initials.setdefault(pair, run["initial_model_sha256"]), run["initial_model_sha256"])
            self.assertEqual(run["seen_transitions"], 12)
            self.assertEqual(run["gradient_updates"], 11)
            warm = run["behavior_variant"] == "greedy_warmup"
            self.assertEqual("training_behavior" in run["training"][0], warm)
            self.assertNotIn("training_behavior", run["training"][1])
            self.assertEqual(run["teacher_action_count"], 6 if warm else 0)
            self.assertEqual(run["training"][1]["training_slots_before"], run["training"][0]["training_slots_after"])
            for episodes in run["validation"].values():
                for episode in episodes:
                    self.assertNotIn("training_behavior", episode)
                    self.assertTrue(all(a["epsilon"] == 0. and "behavior_source" not in a for a in episode["actions"]))

    def test_standard_variant_reproduces_existing_return_scope_runner(self):
        original = run_return_scope_audit(protocol=self.protocol.paired, scenarios=self.scenarios)
        by_key = {(r["return_scope"], r["algorithm"], r["agent_seed"]): r for r in original["runs"]}
        added = {"behavior_variant", "training_behavior_contract", "teacher_action_count", "teacher_completed_at_training_slot"}
        for run in self.report["runs"]:
            if run["behavior_variant"] == "standard":
                stripped = {key: value for key, value in run.items() if key not in added}
                self.assertEqual(stripped, by_key[run["return_scope"], run["algorithm"], run["agent_seed"]])

    def test_checkpoint_identity_records_variant_and_actual_phase_clock(self):
        paired = replace(self.protocol.paired, learning=replace(self.protocol.paired.learning,
            algorithms=("qr_dqn_cvar",), agent_seeds=(11,), validation_profiles=("nominal",), validation_cases=1))
        small = replace(self.protocol, paired=paired)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fresh"
            report = run_behavior_warmup_audit(protocol=small, scenarios=self.scenarios, output_directory=output)
            encoder = ObservationEncoder(max_server_vehicles=2, scales=paired.learning.feature_scales)
            for run in report["runs"]:
                path = output / run["checkpoint"]
                _, meta = load_behavior_checkpoint(path, encoder, protocol=small, variant=run["behavior_variant"],
                    scope=run["return_scope"], algorithm=run["algorithm"], seed=run["agent_seed"])
                self.assertEqual(meta["teacher_action_count"], run["teacher_action_count"])
                self.assertEqual(meta["teacher_completed_at_training_slot"], run["teacher_completed_at_training_slot"])
                wrong = "standard" if run["behavior_variant"] == "greedy_warmup" else "greedy_warmup"
                with self.assertRaises(ValueError):
                    load_behavior_checkpoint(path, encoder, protocol=small, variant=wrong,
                        scope=run["return_scope"], algorithm=run["algorithm"], seed=run["agent_seed"])
                state = torch.load(path, weights_only=True)
                invalid_meta = json.loads(state["run_metadata_json"])
                invalid_meta["behavior_phase_at_checkpoint"] = "teacher"
                state["run_metadata_json"] = json.dumps(invalid_meta)
                bad_path = output / (run["behavior_variant"] + run["return_scope"] + "-bad-phase.pt")
                torch.save(state, bad_path)
                with self.assertRaisesRegex(ValueError, "phase"):
                    load_behavior_checkpoint(bad_path, encoder, protocol=small, variant=run["behavior_variant"],
                        scope=run["return_scope"], algorithm=run["algorithm"], seed=run["agent_seed"])
            saved = (output / "report.json").read_bytes()
            json.loads(saved)
            with self.assertRaises(FileExistsError):
                run_behavior_warmup_audit(protocol=small, scenarios=self.scenarios, output_directory=output)
            self.assertEqual(saved, (output / "report.json").read_bytes())

    def test_validation_and_test_boundaries_are_not_extended_by_teacher(self):
        paired = replace(self.protocol.paired, learning=replace(self.protocol.paired.learning,
            algorithms=("qr_dqn_cvar",), agent_seeds=(11,), validation_profiles=("nominal",), validation_cases=1))
        small = replace(self.protocol, paired=paired)
        original, seen = DevelopmentProtocol.episode_seeds, []

        def guard(instance, split, index):
            self.assertNotEqual(split, "test")
            seen.append(split)
            return original(instance, split, index)

        with patch.object(DevelopmentProtocol, "episode_seeds", guard):
            report = run_behavior_warmup_audit(protocol=small, scenarios=self.scenarios)
        self.assertEqual(set(seen), {"train", "validation"})
        self.assertFalse(report["test_split_accessed"])
        self.assertFalse(report["main_training_started"])
        self.assertFalse(report["centralized_controller"])

    def test_invalid_scenario_binding_fails_before_output_or_generation(self):
        bad_learning = replace(self.protocol.paired.learning, scenario_protocol_sha256="0" * 64)
        bad = replace(self.protocol, paired=replace(self.protocol.paired, learning=bad_learning))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "absent"
            with patch("experiments.run_behavior_warmup_audit.generate_scenario", side_effect=AssertionError("too soon")):
                with self.assertRaises(ValueError):
                    run_behavior_warmup_audit(protocol=bad, scenarios=self.scenarios, output_directory=output)
            self.assertFalse(output.exists())
