"""Matched encoder comparison preserves physical budgets and legacy controls."""

import importlib.util
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if HAS_LEARNING:
    import torch
    from experiments.return_scope_protocol import ReturnScopeProtocol
    from experiments.run_return_scope_audit import _episode, run_return_scope_audit
    from experiments.run_snr_representation_audit import (
        load_snr_representation_checkpoint, run_snr_representation_audit,
    )
    from experiments.snr_representation_protocol import SNRRepresentationProtocol
    from risk_aware_v2x.agents import LogSNRObservationEncoder
    from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol
    from test_return_scope_audit import fixture_plans


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for SNR representation audits")
class SNRRepresentationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        paired, cls.scenarios = fixture_plans()
        cls.learning = replace(paired.learning, agent_seeds=(11,),
                               validation_profiles=("nominal",), validation_cases=1)
        cls.protocol = SNRRepresentationProtocol("snr-fixture", cls.learning)
        cls.output = cls.root / "paired"
        cls.report = run_snr_representation_audit(protocol=cls.protocol, scenarios=cls.scenarios,
                                                  output_directory=cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_grid_matches_initial_weights_and_training_budgets(self):
        runs = self.report["runs"]
        self.assertEqual(len(runs), 4)
        self.assertEqual({(run["representation_variant"], run["algorithm"], run["agent_seed"])
                          for run in runs}, {(variant, algorithm, 11)
                          for variant in self.protocol.variants for algorithm in self.learning.algorithms})
        self.assertEqual(len({run["initial_model_sha256"] for run in runs}), 1)
        self.assertEqual(self.report["training_budget"]["offered_tasks_per_model"], 12)
        self.assertEqual(self.report["training_budget"]["expected_updates_per_model"], 11)
        for run in runs:
            with self.subTest(variant=run["representation_variant"], algorithm=run["algorithm"]):
                variant = run["representation_variant"]
                self.assertEqual(run["return_scope"], "individual")
                self.assertIsNone(run["learning_return_contract"])
                self.assertEqual(run["seen_transitions"], 12)
                self.assertEqual(run["gradient_updates"], 11)
                self.assertEqual(run["target_synchronizations"], 3)
                self.assertEqual(run["representation_contract"], self.protocol.variant_contract(variant))
                self.assertEqual(run["encoder"], self.report["encoders"][variant])
                self.assertEqual(run["encoder"]["schema_version"], 3 if variant == "linear_v3" else 4)
                diagnostics = run["input_diagnostics"]
                for phase in ("initial_network", "final_network"):
                    self.assertEqual(diagnostics[phase]["sample_count"], 12)
                    self.assertTrue(all(diagnostics[phase]["mutation_checks"].values()))

    def test_physical_costs_and_validation_pairing_remain_complete_and_frozen(self):
        self.assertEqual(self.report["physical_episode_executions"], 17)
        self.assertEqual(len(self.report["analytical_validation"]["nominal"]["policies"]), 5)
        for run in self.report["runs"]:
            for episode in run["training"] + run["validation"]["nominal"]:
                self.assertEqual(episode["metrics"]["arrived_tasks"], 6.)
                self.assertEqual(episode["metrics"]["completed_tasks"], 6.)
                self.assertAlmostEqual(math.fsum(episode["accounting"]["total_cost"].values()),
                                       episode["metrics"]["total_cost"])
                self.assertNotIn("training_behavior", episode)
                self.assertTrue(all("behavior_source" not in action for action in episode["actions"]))
            for learned, reference in zip(run["validation"]["nominal"],
                    self.report["analytical_validation"]["nominal"]["scenarios"], strict=True):
                self.assertEqual(learned["seeds"], reference["seeds"])
                self.assertEqual(learned["scenario_fingerprint"], reference["fingerprint"])
                self.assertEqual(learned["updates"], [])
                self.assertEqual(learned["training_slots_before"], run["training_slots"])
                self.assertEqual(learned["training_slots_after"], run["training_slots"])
                self.assertTrue(all(action["epsilon"] == 0. for action in learned["actions"]))
        self.assertFalse(self.report["main_training_started"])
        self.assertFalse(self.report["test_split_accessed"])
        self.assertFalse(self.report["centralized_controller"])

    def test_linear_v3_control_exactly_reproduces_existing_individual_runner(self):
        original = run_return_scope_audit(protocol=ReturnScopeProtocol("snr-control", self.learning),
                                          scenarios=self.scenarios)
        by_key = {(run["algorithm"], run["agent_seed"]): run for run in original["runs"]
                  if run["return_scope"] == "individual"}
        additions = {"representation_variant", "representation_contract", "encoder", "input_diagnostics", "checkpoint"}
        for run in self.report["runs"]:
            if run["representation_variant"] == "linear_v3":
                reference = by_key[run["algorithm"], run["agent_seed"]]
                self.assertEqual({key: value for key, value in run.items() if key not in additions},
                                 {key: value for key, value in reference.items() if key != "checkpoint"})

    def test_checkpoint_metadata_and_encoder_identity_and_cross_variant_rejection(self):
        source = self.report["provenance"]["source_sha256"]
        for run in self.report["runs"]:
            variant, algorithm = run["representation_variant"], run["algorithm"]
            encoder = self.protocol.encoder_for(variant, max_server_vehicles=2)
            path = self.output / run["checkpoint"]
            self.assertEqual(run["checkpoint"], f"checkpoints/{variant}/individual/{algorithm}-seed-11-final.pt")
            agent, metadata = load_snr_representation_checkpoint(path, encoder, protocol=self.protocol,
                variant=variant, algorithm=algorithm, seed=11, source_sha256=source)
            self.assertEqual(agent.seen_transitions, 12)
            self.assertEqual(metadata["representation_variant"], variant)
            self.assertEqual(metadata["representation_contract"], self.protocol.variant_contract(variant))
            self.assertEqual(metadata["snr_representation_protocol_sha256"], self.protocol.fingerprint)
            self.assertEqual(metadata["return_scope"], "individual")
            self.assertEqual(metadata["model_sha256"], run["model_sha256"])
            for wrong in ("log_snr_v4" if variant == "linear_v3" else "linear_v3", "unknown"):
                with self.subTest(variant=variant, wrong=wrong), self.assertRaises(ValueError):
                    load_snr_representation_checkpoint(path, encoder, protocol=self.protocol,
                        variant=wrong, algorithm=algorithm, seed=11, source_sha256=source)
            with self.assertRaises(ValueError):
                load_snr_representation_checkpoint(path,
                    self.protocol.encoder_for(variant, max_server_vehicles=3), protocol=self.protocol,
                    variant=variant, algorithm=algorithm, seed=11, source_sha256=source)
            with self.assertRaises(ValueError):
                load_snr_representation_checkpoint(path, encoder, protocol=self.protocol,
                    variant=variant, algorithm=algorithm, seed=11, source_sha256="0" * 64)
        logarithmic = next(run for run in self.report["runs"] if run["representation_variant"] == "log_snr_v4")
        wrong_reference = LogSNRObservationEncoder(max_server_vehicles=2,
            scales=replace(self.learning.feature_scales, snr_linear=9.))
        with self.assertRaises(ValueError):
            load_snr_representation_checkpoint(self.output / logarithmic["checkpoint"], wrong_reference,
                protocol=self.protocol, variant="log_snr_v4", algorithm=logarithmic["algorithm"], seed=11,
                source_sha256=source)

    def test_copied_representation_metadata_and_modified_actual_weights_are_rejected(self):
        run = self.report["runs"][0]
        variant, algorithm = run["representation_variant"], run["algorithm"]
        encoder = self.protocol.encoder_for(variant, max_server_vehicles=2)
        path = self.output / run["checkpoint"]
        source = self.report["provenance"]["source_sha256"]
        for name in ("representation", "weights"):
            state = torch.load(path, weights_only=True, map_location="cpu")
            if name == "representation":
                metadata = json.loads(state["run_metadata_json"])
                other = "log_snr_v4" if variant == "linear_v3" else "linear_v3"
                metadata["representation_variant"] = other
                metadata["representation_contract"] = self.protocol.variant_contract(other)
                state["run_metadata_json"] = json.dumps(metadata)
            else:
                next(iter(state["online"].values())).add_(1.)
            invalid = self.root / f"wrong-{name}.pt"
            torch.save(state, invalid)
            with self.subTest(tampered=name), self.assertRaises(ValueError):
                load_snr_representation_checkpoint(invalid, encoder, protocol=self.protocol,
                    variant=variant, algorithm=algorithm, seed=11, source_sha256=source)

    def test_only_train_and_validation_seed_keys_are_requested(self):
        original, requested = DevelopmentProtocol.episode_seeds, []

        def guard(instance, split, index):
            self.assertNotEqual(split, "test")
            requested.append(split)
            return original(instance, split, index)

        with patch.object(DevelopmentProtocol, "episode_seeds", guard):
            report = run_snr_representation_audit(protocol=self.protocol, scenarios=self.scenarios,
                                                  output_directory=self.root / "split-boundary")
        self.assertEqual(set(requested), {"train", "validation"})
        self.assertFalse(report["test_split_accessed"])

    def test_existing_output_is_not_overwritten_and_invalid_plan_never_generates_scenes(self):
        saved = (self.output / "report.json").read_bytes()
        with self.assertRaises(FileExistsError):
            run_snr_representation_audit(protocol=self.protocol, scenarios=self.scenarios,
                                          output_directory=self.output)
        self.assertEqual(saved, (self.output / "report.json").read_bytes())
        bad = replace(self.protocol, learning=replace(self.learning, scenario_protocol_sha256="0" * 64))
        destination = self.root / "absent-invalid"
        with patch("experiments.run_snr_representation_audit.generate_scenario", side_effect=AssertionError("too early")):
            with self.assertRaises(ValueError):
                run_snr_representation_audit(protocol=bad, scenarios=self.scenarios, output_directory=destination)
        self.assertFalse(destination.exists())

    def test_frozen_validation_state_check_rejects_mutation(self):
        def mutate(*args, **kwargs):
            report, clock = _episode(*args, **kwargs)
            if not kwargs["training"]:
                args[3].gradient_updates += 1
            return report, clock

        with patch("experiments.run_return_scope_audit._episode", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "mutated"):
                run_snr_representation_audit(protocol=self.protocol, scenarios=self.scenarios,
                                              output_directory=self.root / "mutation")

    def test_variant_and_algorithm_iteration_order_do_not_change_model_runs(self):
        reversed_plan = replace(self.protocol, variants=self.protocol.variants[::-1],
                                learning=replace(self.learning, algorithms=self.learning.algorithms[::-1]))
        reversed_report = run_snr_representation_audit(protocol=reversed_plan, scenarios=self.scenarios,
                                                       output_directory=self.root / "reordered")
        keyed = lambda runs: {(run["representation_variant"], run["algorithm"], run["agent_seed"]): run for run in runs}
        self.assertEqual(keyed(reversed_report["runs"]), keyed(self.report["runs"]))


if __name__ == "__main__":
    unittest.main()
