"""Frozen representation controls, strict JSON, and explicit encoder contracts."""

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
    from experiments.return_scope_protocol import ReturnScopeProtocol
    from experiments.snr_representation_protocol import DEFAULT_SNR_REPRESENTATION_PROTOCOL, SNRRepresentationProtocol
    from risk_aware_v2x.agents import ObservationEncoder
    from risk_aware_v2x.agents.encoding import LogSNRObservationEncoder
    from risk_aware_v2x.agents.neural import encoder_signature
    from risk_aware_v2x.scenarios.generator import canonical_json
    from risk_aware_v2x.scenarios.protocol import DevelopmentProtocol


@unittest.skipUnless(HAS_LEARNING, "install learning dependencies for representation protocol tests")
class SNRRepresentationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = DEFAULT_SNR_REPRESENTATION_PROTOCOL
        cls.plan = SNRRepresentationProtocol.load(cls.path)
        cls.scenarios = DevelopmentProtocol.load(cls.path.with_name("development_v2.json"))
        cls.record = json.loads(cls.path.read_text(encoding="utf-8"))

    def load_record(self, record):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
            return SNRRepresentationProtocol.load(path)

    def test_new_plan_preserves_exact_existing_paired_learning_budget(self):
        original_path = self.path.with_name("return_scope_pilot_v1.json")
        original_bytes = original_path.read_bytes()
        original = ReturnScopeProtocol.load(original_path)
        self.plan.validate_against(self.scenarios)
        self.assertEqual(self.plan.learning, original.learning)
        self.assertEqual(self.plan.variants, ("linear_v3", "log_snr_v4"))
        self.assertEqual(self.plan.return_scope, "individual")
        self.assertEqual(self.plan.learning.training_episodes, 20)
        self.assertEqual(self.plan.learning.validation_cases, 4)
        self.assertEqual(self.plan.learning.agent_seeds, (11, 29))
        self.assertEqual(self.plan.learning.algorithms, ("qr_dqn", "qr_dqn_cvar"))
        self.assertEqual(original_path.read_bytes(), original_bytes)

    def test_actual_encoders_differ_but_shared_template_remains_legacy(self):
        linear = self.plan.encoder_for("linear_v3", max_server_vehicles=4)
        logarithmic = self.plan.encoder_for("log_snr_v4", max_server_vehicles=4)
        self.assertIs(type(linear), ObservationEncoder)
        self.assertIs(type(logarithmic), LogSNRObservationEncoder)
        self.assertEqual((linear.schema_version, logarithmic.schema_version), (3, 4))
        self.assertEqual(linear.dimension, logarithmic.dimension)
        self.assertEqual(linear.dimension, 96)
        self.assertEqual(self.plan.learning.encoder_schema_version, 3)
        self.assertNotEqual(encoder_signature(linear), encoder_signature(logarithmic))
        for variant, encoder in (("linear_v3", linear), ("log_snr_v4", logarithmic)):
            contract = self.plan.variant_contract(variant)
            self.assertEqual(contract["representation_variant"], variant)
            self.assertEqual(contract["encoder_schema_version"], encoder.schema_version)
            self.assertEqual(contract["reference_snr_linear"], 100.)
            self.assertIs(contract["fitted"], False)
            self.assertIs(contract["clipped"], False)
        log = self.plan.variant_contract("log_snr_v4")
        self.assertTrue(all(log[key] == value for key, value in logarithmic.representation_contract().items()))
        self.assertEqual(self.plan.variant_contract("linear_v3")["formula"], "snr_linear/reference_snr_linear")
        with self.assertRaises(ValueError):
            self.plan.encoder_for("unknown", max_server_vehicles=4)

    def test_immutable_plan_and_canonical_fingerprint(self):
        with self.assertRaises(FrozenInstanceError):
            self.plan.revision = "changed"
        expected = hashlib.sha256(canonical_json(asdict(self.plan)).encode()).hexdigest()
        self.assertEqual(self.plan.fingerprint, expected)
        self.assertEqual(self.plan.fingerprint, SNRRepresentationProtocol.load(self.path).fingerprint)
        reversed_plan = replace(self.plan, variants=tuple(reversed(self.plan.variants)))
        reversed_plan.validate_against(self.scenarios)
        self.assertNotEqual(reversed_plan.fingerprint, self.plan.fingerprint)
        changed = replace(self.plan, learning=replace(self.plan.learning, training_episodes=19))
        self.assertNotEqual(changed.fingerprint, self.plan.fingerprint)

    def test_variant_scope_version_and_learning_type_are_strict(self):
        for changes in ({"variants": ("linear_v3",)}, {"variants": ("linear_v3", "linear_v3")},
                        {"variants": ["linear_v3", "log_snr_v4"]}, {"variants": ("linear_v3", "other")},
                        {"variants": (True, "log_snr_v4")}, {"return_scope": "team"},
                        {"return_scope": False}, {"schema_version": True}, {"schema_version": 2},
                        {"purpose": "main-training"}, {"revision": " "}, {"learning": {}}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                replace(self.plan, **changes)

    def test_only_declared_qr_pair_and_fixed_method_and_reference_are_allowed(self):
        for algorithms in (("qr_dqn",), ("dqn", "qr_dqn_cvar"), ("double_dqn", "qr_dqn")):
            with self.subTest(algorithms=algorithms), self.assertRaises(ValueError):
                replace(self.plan, learning=replace(self.plan.learning, algorithms=algorithms))
        reverse = replace(self.plan, learning=replace(self.plan.learning, algorithms=("qr_dqn_cvar", "qr_dqn")))
        reverse.validate_against(self.scenarios)
        for changes in ({"gamma_per_slot": .9}, {"alpha": .8}, {"risk_weight": .25}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                common = replace(self.plan.learning.common_agent, **changes)
                replace(self.plan, learning=replace(self.plan.learning, common_agent=common))
        with self.assertRaises(ValueError):
            scales = replace(self.plan.learning.feature_scales, snr_linear=1e6)
            replace(self.plan, learning=replace(self.plan.learning, feature_scales=scales))

    def test_scenario_binding_profiles_seed_and_budget_limits_hold(self):
        for changes in ({"scenario_protocol_sha256": "0" * 64}, {"agent_seeds": (987,)},
                        {"training_profile": "unknown"}, {"validation_profiles": ("unknown",)},
                        {"training_episodes": 21}, {"validation_cases": 7}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.plan, learning=replace(self.plan.learning, **changes)).validate_against(self.scenarios)

    def test_json_unknown_missing_or_wrong_type_fields_reject(self):
        mutations = [lambda r: r.update(teacher_episodes=2), lambda r: r.pop("return_scope"),
                     lambda r: r.update(variants="linear_v3"), lambda r: r.update(schema_version=1.),
                     lambda r: r.update(learning="learning_pilot_v3.json"),
                     lambda r: r["learning"].update(per_variant_budgets={"log_snr_v4": 40}),
                     lambda r: r["learning"]["common_agent"].update(algorithm="qr_dqn"),
                     lambda r: r["learning"].update(encoder_schema_version=4)]
        for mutate in mutations:
            record = copy.deepcopy(self.record)
            mutate(record)
            with self.assertRaises((ValueError, TypeError)):
                self.load_record(record)

    def test_duplicate_json_keys_and_other_protocols_are_rejected(self):
        original = self.path.read_text(encoding="utf-8")
        duplicate = original.replace('"training_episodes": 20', '"training_episodes": 20, "training_episodes": 20')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                SNRRepresentationProtocol.load(path)
        for filename in ("learning_pilot_v3.json", "return_scope_pilot_v1.json", "behavior_warmup_pilot_v1.json"):
            path = self.path.with_name(filename)
            before = path.read_bytes()
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                SNRRepresentationProtocol.load(path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
