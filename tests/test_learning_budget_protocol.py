"""Frozen budget counts, expanded seed prefixes and unchanged scientific controls."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from unittest.mock import patch

HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if HAS_LEARNING:
    from experiments.learning_budget_protocol import (
        DEFAULT_BUDGET_SCENARIO_PROTOCOL, DEFAULT_LEARNING_BUDGET_PROTOCOL,
        FIXED_POLICY_NAMES, LearningBudgetProtocol,
    )
    from experiments.snr_representation_protocol import DEFAULT_SNR_REPRESENTATION_PROTOCOL, SNRRepresentationProtocol
    from risk_aware_v2x.agents import LogSNRObservationEncoder, ObservationEncoder
    from risk_aware_v2x.agents.neural import NeuralCostAgent, encoder_signature
    from risk_aware_v2x.scenarios.generator import canonical_json
    from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for budget protocol controls")
class LearningBudgetProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = DEFAULT_LEARNING_BUDGET_PROTOCOL
        cls.plan = LearningBudgetProtocol.load(cls.path)
        cls.scenarios = DevelopmentProtocol.load(DEFAULT_BUDGET_SCENARIO_PROTOCOL)
        cls.old_scenarios = DevelopmentProtocol.load(cls.path.with_name("development_v2.json"))
        cls.old_controls = SNRRepresentationProtocol.load(DEFAULT_SNR_REPRESENTATION_PROTOCOL)
        cls.record = json.loads(cls.path.read_text(encoding="utf-8"))

    def load_record(self, record):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            path.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
            return LearningBudgetProtocol.load(path)

    def test_expanded_plan_preserves_physics_and_existing_seed_prefixes(self):
        old, new = self.old_scenarios, self.scenarios
        self.assertEqual(new.profiles, old.profiles)
        self.assertEqual(new.master_seed, old.master_seed)
        self.assertEqual(new.agent_seeds, old.agent_seeds)
        self.assertEqual(new.cvar_alpha, old.cvar_alpha)
        self.assertEqual(new.split_counts, (500, 20, 30))
        self.assertNotEqual(new.fingerprint, old.fingerprint)
        for split, count in (("train", 20), ("validation", 6)):
            for index in range(count):
                self.assertEqual(new.episode_seeds(split, index), old.episode_seeds(split, index))
        added = [new.episode_seeds("train", index) for index in range(20, 500)]
        self.assertEqual(len({key.scenario_seed for key in added}), 480)
        self.assertNotEqual(added[0].scenario_seed, new.episode_seeds("validation", 0).scenario_seed)

    def test_learning_changes_only_trace_budget_identity_and_declared_seed_axis(self):
        old, new = asdict(self.old_controls.learning), asdict(self.plan.learning)
        changes = {name for name in old if old[name] != new[name]}
        self.assertEqual(changes, {"revision", "scenario_protocol_sha256", "training_episodes",
                                   "validation_cases", "agent_seeds"})
        self.assertEqual(self.plan.learning.agent_seeds, (11, 29, 47))
        self.assertEqual(self.plan.learning.training_episodes, 500)
        self.assertEqual(self.plan.learning.validation_cases, 20)
        self.assertEqual(self.plan.learning.scenario_protocol_sha256, self.scenarios.fingerprint)
        self.plan.validate_against(self.scenarios)

    def test_predeclared_execution_counts_do_not_repeat_training_or_fixed_policies(self):
        plan = self.plan.execution_plan()
        self.assertEqual(plan["model_chains"], 2 * 2 * 3)
        self.assertEqual(plan["training_episode_executions"], 12 * 500)
        self.assertEqual(plan["learned_validation_episode_executions"], 12 * 3 * 4 * 20)
        self.assertEqual(plan["fixed_policy_episode_executions"], 5 * 4 * 20)
        self.assertEqual(plan["total_physical_episode_executions"], 9280)
        self.assertEqual(plan["validation_episodes_per_budget_per_chain"], 80)
        self.assertEqual(plan["fixed_policy_names"], list(FIXED_POLICY_NAMES))
        self.assertEqual(plan["budgets"], [20, 100, 500])
        self.assertTrue(plan["planning_only"])
        self.assertTrue(plan["fixed_validation_cohort_across_budgets"])
        self.assertFalse(plan["main_training_started"])
        self.assertFalse(plan["test_split_accessed"])

    def test_next_budget_uses_strictly_later_boundary_and_valid_counts(self):
        for completed, expected in ((0, 20), (19, 20), (20, 100), (99, 100),
                                    (100, 500), (499, 500), (500, None)):
            self.assertEqual(self.plan.next_budget(completed), expected)
        for completed in (-1, True, 1., 501, "20"):
            with self.subTest(completed=completed), self.assertRaises((ValueError, TypeError)):
                self.plan.next_budget(completed)

    def test_encoder_contracts_match_existing_versioned_representation_controls(self):
        for variant, cls in (("linear_v3", ObservationEncoder), ("log_snr_v4", LogSNRObservationEncoder)):
            encoder = self.plan.encoder_for(variant, max_server_vehicles=4)
            self.assertIs(type(encoder), cls)
            self.assertEqual(encoder_signature(encoder), encoder_signature(
                self.old_controls.encoder_for(variant, max_server_vehicles=4)))
            self.assertEqual(self.plan.variant_contract(variant), self.old_controls.variant_contract(variant))
            self.assertFalse(self.plan.variant_contract(variant)["fitted"])
            self.assertFalse(self.plan.variant_contract(variant)["clipped"])
        with self.assertRaises(ValueError):
            self.plan.encoder_for("other", max_server_vehicles=4)
        self.assertNotEqual(self.plan.fingerprint, self.old_controls.fingerprint)

    def test_planning_and_validation_never_construct_scenes_or_train(self):
        forbidden = AssertionError("planning must not generate scenes or train")
        requested = []
        original = DevelopmentProtocol.episode_seeds

        def guard(instance, split, index):
            self.assertNotEqual(split, "test")
            requested.append(split)
            return original(instance, split, index)

        with patch("risk_aware_v2x.scenarios.generate_scenario", side_effect=forbidden), \
                patch.object(NeuralCostAgent, "__init__", side_effect=forbidden), \
                patch.object(DevelopmentProtocol, "episode_seeds", guard):
            self.plan.validate_against(self.scenarios)
            execution = self.plan.execution_plan()
        self.assertEqual(set(requested), {"train", "validation"})
        self.assertEqual(execution["total_physical_episode_executions"], 9280)

    def test_fingerprint_and_immutability_bind_all_budget_controls(self):
        self.assertEqual(self.plan.fingerprint,
                         hashlib.sha256(canonical_json(asdict(self.plan)).encode()).hexdigest())
        self.assertEqual(self.plan.fingerprint, LearningBudgetProtocol.load(self.path).fingerprint)
        with self.assertRaises(FrozenInstanceError):
            self.plan.budgets = (10, 500)
        changed = replace(self.plan, budgets=(10, 100, 500))
        self.assertNotEqual(changed.fingerprint, self.plan.fingerprint)
        changed = replace(self.plan, checkpoint_interval_episodes=5)
        self.assertNotEqual(changed.fingerprint, self.plan.fingerprint)

    def test_strict_budgets_versions_checkpoint_rules_and_scope(self):
        for budgets in ((), [20, 100, 500], (0, 500), (True, 500), (20., 500),
                        (20, 20, 500), (100, 20, 500), (20, 100, 499)):
            with self.subTest(budgets=budgets), self.assertRaises((ValueError, TypeError)):
                replace(self.plan, budgets=budgets)
        for changes in ({"checkpoint_interval_episodes": 0}, {"checkpoint_interval_episodes": True},
                        {"checkpoint_interval_episodes": 1.}, {"schema_version": 2},
                        {"schema_version": True}, {"revision": " "}, {"learning": {}},
                        {"purpose": "main-training"}, {"return_scope": "team"},
                        {"selection_rule": "best-validation-cost"}, {"variants": ("log_snr_v4",)}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                replace(self.plan, **changes)

    def test_tiny_declared_fixture_and_original_algorithm_controls(self):
        learning = replace(self.plan.learning, training_episodes=2, validation_cases=1, agent_seeds=(11,))
        tiny = replace(self.plan, learning=learning, budgets=(1, 2), checkpoint_interval_episodes=10)
        self.assertEqual(tiny.execution_plan()["model_chains"], 4)
        self.assertEqual(tiny.execution_plan()["training_episode_executions"], 8)
        self.assertEqual(tiny.execution_plan()["learned_validation_episode_executions"], 32)
        for field, value in (("gamma_per_slot", .9), ("alpha", .8), ("risk_weight", .25)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(self.plan, learning=replace(self.plan.learning,
                    common_agent=replace(self.plan.learning.common_agent, **{field: value})))
        with self.assertRaises(ValueError):
            replace(self.plan, learning=replace(self.plan.learning,
                feature_scales=replace(self.plan.learning.feature_scales, snr_linear=1e6)))
        with self.assertRaises(ValueError):
            replace(self.plan, learning=replace(self.plan.learning, algorithms=("qr_dqn_cvar",)))

    def test_scenario_binding_capacity_profiles_and_seeds_fail_closed(self):
        for changes in ({"scenario_protocol_sha256": "0" * 64}, {"agent_seeds": (987,)},
                        {"training_profile": "missing"}, {"validation_profiles": ("missing",)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.plan, learning=replace(self.plan.learning, **changes)).validate_against(self.scenarios)
        with self.assertRaises(ValueError):
            self.plan.validate_against(replace(self.scenarios, split_counts=(499, 20, 30)))
        with self.assertRaises(ValueError):
            self.plan.validate_against(self.old_scenarios)

    def test_strict_json_missing_unknown_duplicate_and_wrong_types(self):
        mutations = [lambda r: r.pop("budgets"), lambda r: r.update(teacher_episodes=2),
                     lambda r: r.update(budgets="20,100,500"), lambda r: r.update(variants="linear_v3"),
                     lambda r: r.update(schema_version=1.), lambda r: r.update(learning="path.json"),
                     lambda r: r.update(selection_rule="best-validation"),
                     lambda r: r["learning"]["common_agent"].update(algorithm="qr_dqn")]
        for mutate in mutations:
            record = copy.deepcopy(self.record)
            mutate(record)
            with self.assertRaises((ValueError, TypeError)):
                self.load_record(record)
        text = self.path.read_text(encoding="utf-8")
        duplicate = text.replace('"checkpoint_interval_episodes": 10',
                                 '"checkpoint_interval_episodes": 10, "checkpoint_interval_episodes": 10')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                LearningBudgetProtocol.load(path)
        for name in ("snr_representation_pilot_v1.json", "learning_pilot_v3.json", "return_scope_pilot_v1.json"):
            path = self.path.with_name(name)
            before = path.read_bytes()
            with self.subTest(name=name), self.assertRaises(ValueError):
                LearningBudgetProtocol.load(path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
