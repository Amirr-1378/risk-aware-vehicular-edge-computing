"""Frozen paired scope budgets, strict JSON contracts and scenario binding."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

HAS_LEARNING = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))
if HAS_LEARNING:
    from experiments.return_scope_protocol import DEFAULT_RETURN_SCOPE_PROTOCOL, ReturnScopeProtocol
    from experiments.learning_protocol import LearningProtocol
    from risk_aware_v2x.scenarios.generator import canonical_json
    from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for return-scope protocol tests")
class ReturnScopeProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = ReturnScopeProtocol.load(DEFAULT_RETURN_SCOPE_PROTOCOL)
        cls.scenarios = DevelopmentProtocol.load(DEFAULT_RETURN_SCOPE_PROTOCOL.with_name("development_v2.json"))
        cls.record = json.loads(DEFAULT_RETURN_SCOPE_PROTOCOL.read_text(encoding="utf-8"))

    def load_record(self, record):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
            return ReturnScopeProtocol.load(path)

    def test_checked_in_plan_preserves_shared_hyperparameters_and_declares_complete_grid(self):
        self.plan.validate_against(self.scenarios)
        learning = self.plan.learning
        original = LearningProtocol.load(DEFAULT_RETURN_SCOPE_PROTOCOL.with_name("learning_pilot_v3.json"))
        self.assertEqual(self.plan.return_scopes, ("individual", "team"))
        self.assertEqual(learning.algorithms, ("qr_dqn", "qr_dqn_cvar"))
        self.assertEqual(learning.agent_seeds, (11, 29))
        self.assertEqual(learning.training_episodes, 20)
        self.assertEqual(learning.validation_cases, 4)
        self.assertEqual(learning.validation_profiles, original.validation_profiles)
        self.assertEqual(learning.common_agent, original.common_agent)
        self.assertEqual(learning.epsilon_schedule, original.epsilon_schedule)
        self.assertEqual(learning.feature_scales, original.feature_scales)
        self.assertEqual(learning.selection_rule, "fixed-final-training-episode")
        self.assertEqual(learning.scenario_protocol_sha256, self.scenarios.fingerprint)

    def test_dataclass_and_nested_learning_are_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.plan.revision = "changed"
        with self.assertRaises(FrozenInstanceError):
            self.plan.learning.training_episodes = 1

    def test_full_fingerprint_is_canonical_and_changes_with_scope_order_or_budget(self):
        expected = hashlib.sha256(canonical_json(asdict(self.plan)).encode("utf-8")).hexdigest()
        self.assertEqual(self.plan.fingerprint, expected)
        self.assertEqual(self.plan.fingerprint, ReturnScopeProtocol.load(DEFAULT_RETURN_SCOPE_PROTOCOL).fingerprint)
        changed = replace(self.plan, learning=replace(self.plan.learning, training_episodes=19))
        self.assertNotEqual(changed.fingerprint, self.plan.fingerprint)
        reversed_scopes = replace(self.plan, return_scopes=("team", "individual"))
        reversed_scopes.validate_against(self.scenarios)
        self.assertNotEqual(reversed_scopes.fingerprint, self.plan.fingerprint)

    def test_incomplete_duplicate_unknown_or_mutable_scope_axes_are_rejected(self):
        for scopes in (("individual",), ("team", "team"), ("individual", "unknown"),
                       ("individual", "team", "team"), ["individual", "team"], (), (False, "team")):
            with self.subTest(scopes=scopes), self.assertRaises((ValueError, TypeError)):
                replace(self.plan, return_scopes=scopes)
        for changes in ({"revision": ""}, {"schema_version": True}, {"schema_version": 2},
                        {"purpose": "main-training"}, {"learning": asdict(self.plan.learning)}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                replace(self.plan, **changes)

    def test_control_contract_rejects_changed_discount_alpha_or_risk_weight(self):
        for changes in ({"gamma_per_slot": .9}, {"alpha": .8}, {"risk_weight": 0.}):
            common = replace(self.plan.learning.common_agent, **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.plan, learning=replace(self.plan.learning, common_agent=common))

    def test_scenario_mismatch_undeclared_seed_profile_and_excessive_budgets_fail(self):
        for changes in ({"scenario_protocol_sha256": "0" * 64}, {"agent_seeds": (987,)},
                        {"training_profile": "unknown"}, {"validation_profiles": ("unknown",)},
                        {"training_episodes": 21}, {"validation_cases": 7}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.plan, learning=replace(self.plan.learning, **changes)).validate_against(self.scenarios)
        for changes in ({"training_episodes": 0}, {"validation_cases": True}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                replace(self.plan.learning, **changes)

    def test_old_plans_are_rejected_without_silent_upgrade_or_file_changes(self):
        for name in ("learning_pilot_v1.json", "learning_pilot_v2.json", "learning_pilot_v3.json", "development_v2.json"):
            path = DEFAULT_RETURN_SCOPE_PROTOCOL.with_name(name)
            before = path.read_bytes()
            with self.subTest(name=name), self.assertRaises(ValueError):
                ReturnScopeProtocol.load(path)
            self.assertEqual(before, path.read_bytes())

    def test_unknown_outer_nested_and_agent_keys_are_rejected(self):
        mutations = [lambda r: r.update(allow_test=True),
                     lambda r: r["learning"].update(per_scope_budgets={"team": 1}),
                     lambda r: r["learning"]["common_agent"].update(algorithm="dqn")]
        for mutate in mutations:
            record = copy.deepcopy(self.record)
            mutate(record)
            with self.assertRaises((TypeError, ValueError)):
                self.load_record(record)

    def test_duplicate_json_keys_are_rejected_at_any_depth(self):
        original = DEFAULT_RETURN_SCOPE_PROTOCOL.read_text(encoding="utf-8")
        for text in ('{"schema_version":1,"schema_version":1}',
                     original.replace('"training_episodes": 20', '"training_episodes": 20, "training_episodes": 20'),
                     original.replace('"alpha": 0.9', '"alpha": 0.9, "alpha": 0.9')):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "duplicate.json"
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    ReturnScopeProtocol.load(path)

    def test_nested_versions_and_json_axis_types_are_strict(self):
        mutations = [lambda r: r["learning"].update(schema_version=2),
                     lambda r: r["learning"].update(cost_model_version=2.),
                     lambda r: r["learning"].update(algorithms="qr_dqn"),
                     lambda r: r.update(return_scopes="individual"),
                     lambda r: r.update(learning="learning_pilot_v3.json")]
        for mutate in mutations:
            record = copy.deepcopy(self.record)
            mutate(record)
            with self.assertRaises((ValueError, TypeError)):
                self.load_record(record)
