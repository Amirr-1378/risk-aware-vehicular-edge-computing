"""Strict behavior-only warmup plans, immutable shared budgets and scope controls."""

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
    from experiments.behavior_warmup_protocol import BehaviorWarmupProtocol, DEFAULT_BEHAVIOR_WARMUP_PROTOCOL
    from experiments.return_scope_protocol import ReturnScopeProtocol
    from risk_aware_v2x.scenarios.generator import canonical_json
    from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for behavior-warmup protocol tests")
class BehaviorWarmupProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = DEFAULT_BEHAVIOR_WARMUP_PROTOCOL
        cls.plan = BehaviorWarmupProtocol.load(cls.path)
        cls.record = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.scenarios = DevelopmentProtocol.load(cls.path.with_name("development_v2.json"))

    def load_record(self, record):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
            return BehaviorWarmupProtocol.load(path)

    def test_checked_in_plan_reuses_exact_paired_plan_without_extra_training_budget(self):
        original_path = self.path.with_name("return_scope_pilot_v1.json")
        paired = ReturnScopeProtocol.load(original_path)
        self.plan.validate_against(self.scenarios)
        self.assertEqual(self.plan.paired, paired)
        self.assertEqual(self.record["paired"], json.loads(original_path.read_text(encoding="utf-8")))
        self.assertEqual(self.plan.teacher_episodes, 2)
        self.assertEqual(self.plan.variants, ("standard", "greedy_warmup"))
        self.assertEqual(self.plan.paired.learning.training_episodes, 20)
        self.assertEqual(self.plan.paired.learning.validation_cases, 4)
        self.assertEqual(self.plan.paired.learning.algorithms, ("qr_dqn", "qr_dqn_cvar"))
        self.assertEqual(self.plan.paired.learning.agent_seeds, (11, 29))
        self.assertEqual(self.plan.paired.learning.common_agent.alpha, .9)
        self.assertEqual(self.plan.paired.learning.common_agent.risk_weight, .5)
        self.assertEqual(self.plan.paired.learning.common_agent.gamma_per_slot, 1.)

    def test_plan_and_nested_templates_are_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.plan.teacher_episodes = 3
        with self.assertRaises(FrozenInstanceError):
            self.plan.paired.learning.training_episodes = 30

    def test_fingerprint_is_full_canonical_and_changes_with_behavior_or_budget(self):
        expected = hashlib.sha256(canonical_json(asdict(self.plan)).encode("utf-8")).hexdigest()
        self.assertEqual(self.plan.fingerprint, expected)
        self.assertEqual(self.plan.fingerprint, BehaviorWarmupProtocol.load(self.path).fingerprint)
        self.assertNotEqual(self.plan.fingerprint, replace(self.plan, teacher_episodes=1).fingerprint)
        self.assertNotEqual(self.plan.fingerprint, replace(self.plan, variants=self.plan.variants[::-1]).fingerprint)
        paired = replace(self.plan.paired, learning=replace(self.plan.paired.learning, training_episodes=19))
        self.assertNotEqual(self.plan.fingerprint, replace(self.plan, paired=paired).fingerprint)

    def test_variants_must_be_complete_unique_and_immutable(self):
        for variants in (("standard",), ("greedy_warmup", "greedy_warmup"),
                         ("standard", "unknown"), ("standard", "greedy_warmup", "standard"),
                         ["standard", "greedy_warmup"], (), (False, "greedy_warmup")):
            with self.subTest(variants=variants), self.assertRaises((TypeError, ValueError)):
                replace(self.plan, variants=variants)
        reverse = replace(self.plan, variants=self.plan.variants[::-1])
        reverse.validate_against(self.scenarios)

    def test_teacher_budget_is_strictly_positive_and_leaves_learner_behavior(self):
        for count in (0, -1, True, 1.5, 20, 21):
            with self.subTest(count=count), self.assertRaises((TypeError, ValueError)):
                replace(self.plan, teacher_episodes=count)
        short = replace(self.plan.paired, learning=replace(self.plan.paired.learning, training_episodes=2))
        self.assertEqual(replace(self.plan, paired=short, teacher_episodes=1).teacher_episodes, 1)
        with self.assertRaises(ValueError):
            replace(self.plan, paired=short)

    def test_bad_outer_versions_purpose_revision_or_nested_type_are_rejected(self):
        for changes in ({"schema_version": True}, {"schema_version": 2}, {"purpose": "main-training"},
                        {"revision": ""}, {"paired": asdict(self.plan.paired)}):
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                replace(self.plan, **changes)

    def test_scenario_binding_and_selected_split_budgets_are_preserved(self):
        for changes in ({"scenario_protocol_sha256": "0" * 64}, {"agent_seeds": (987,)},
                        {"training_profile": "unknown"}, {"validation_profiles": ("unknown",)},
                        {"training_episodes": 21}, {"validation_cases": 7}):
            learning = replace(self.plan.paired.learning, **changes)
            plan = replace(self.plan, paired=replace(self.plan.paired, learning=learning))
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                plan.validate_against(self.scenarios)

    def test_unchanged_qr_cvar_control_contract_remains_enforced(self):
        for changes in ({"gamma_per_slot": .9}, {"alpha": .8}, {"risk_weight": 0.}):
            record = copy.deepcopy(self.record)
            record["paired"]["learning"]["common_agent"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load_record(record)

    def test_historical_plans_are_not_silently_upgraded_or_modified(self):
        for name in ("return_scope_pilot_v1.json", "learning_pilot_v1.json", "learning_pilot_v2.json",
                     "learning_pilot_v3.json", "development_v2.json"):
            path = self.path.with_name(name)
            before = path.read_bytes()
            with self.subTest(name=name), self.assertRaises(ValueError):
                BehaviorWarmupProtocol.load(path)
            self.assertEqual(before, path.read_bytes())

    def test_unknown_fields_missing_fields_and_bad_json_types_are_rejected(self):
        mutations = [lambda r: r.update(extra_updates=10), lambda r: r.pop("teacher_episodes"),
            lambda r: r.pop("variants"), lambda r: r.update(variants="standard"),
            lambda r: r.update(paired="return_scope_pilot_v1.json"),
            lambda r: r["paired"].update(teacher_policy="oracle"),
            lambda r: r["paired"]["learning"].update(per_variant_budgets={"standard": 20}),
            lambda r: r["paired"]["learning"]["common_agent"].update(algorithm="dqn")]
        for mutate in mutations:
            record = copy.deepcopy(self.record)
            mutate(record)
            with self.assertRaises((TypeError, ValueError)):
                self.load_record(record)

    def test_nested_versions_scopes_and_axis_types_are_strict(self):
        mutations = [lambda r: r["paired"].update(schema_version=True),
            lambda r: r["paired"].update(purpose="main-training"),
            lambda r: r["paired"].pop("return_scopes"),
            lambda r: r["paired"].update(return_scopes="individual"),
            lambda r: r["paired"].update(return_scopes=["team", "team"]),
            lambda r: r["paired"]["learning"].update(schema_version=2),
            lambda r: r["paired"]["learning"].update(cost_model_version=2.),
            lambda r: r["paired"]["learning"].update(algorithms="qr_dqn")]
        for mutate in mutations:
            record = copy.deepcopy(self.record)
            mutate(record)
            with self.assertRaises((TypeError, ValueError)):
                self.load_record(record)

    def test_duplicate_json_keys_are_rejected_at_outer_and_nested_depth(self):
        original = self.path.read_text(encoding="utf-8")
        texts = ['{"schema_version":1,"schema_version":1}',
            original.replace('"teacher_episodes": 2', '"teacher_episodes": 2, "teacher_episodes": 2'),
            original.replace('"training_episodes": 20', '"training_episodes": 20, "training_episodes": 20'),
            original.replace('"alpha": 0.9', '"alpha": 0.9, "alpha": 0.9')]
        for text in texts:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "duplicate.json"
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    BehaviorWarmupProtocol.load(path)
