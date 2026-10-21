"""Arrival, resource, mobility-envelope, and RNG isolation scientific fixtures."""

import json
import math
import random
import unittest
from dataclasses import asdict, replace

from risk_aware_v2x.entities import Criticality
from risk_aware_v2x.models.mobility import MotionState, advance_straight_line
from risk_aware_v2x.scenarios import ScenarioSpec, TaskClassSpec, generate_scenario
from risk_aware_v2x.scenarios.generator import arrival_probability


class StochasticScenarioTests(unittest.TestCase):
    def test_bernoulli_arrival_probability_has_rate_and_time_units(self):
        self.assertEqual(arrival_probability(0., .01), 0.)
        self.assertAlmostEqual(arrival_probability(3., .01), .03)
        self.assertEqual(arrival_probability(100., .01), 1.)
        for rate, duration in ((101., .01), (-1., .01), (1., 0.), (float("nan"), .01)):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                arrival_probability(rate, duration)

    def test_independent_workload_mean_gives_nominal_load_not_mm1_claim(self):
        scene = generate_scenario(ScenarioSpec(), seed=7)
        self.assertAlmostEqual(scene.manifest()["arrivals"]["uncapped_nominal_mec_cpu_load_ratio"], 4 / 15)
        high = generate_scenario(replace(ScenarioSpec(), arrival_rate_per_user_hz=12.), seed=7)
        self.assertAlmostEqual(high.manifest()["arrivals"]["uncapped_nominal_mec_cpu_load_ratio"], 16 / 15)

    def test_equal_seeds_reproduce_every_manifest_without_global_rng_mutation(self):
        before = random.getstate()
        one = generate_scenario(ScenarioSpec(), seed=17)
        two = generate_scenario(ScenarioSpec(), seed=17)
        self.assertEqual(one.manifest(), two.manifest())
        self.assertEqual(one.fingerprint, two.fingerprint)
        self.assertEqual(before, random.getstate())
        json.dumps(one.manifest(), allow_nan=False)

    def test_different_seed_changes_actual_geometry_and_workload(self):
        one, two = (generate_scenario(ScenarioSpec(), seed=seed) for seed in (3, 4))
        self.assertNotEqual(one.fingerprint, two.fingerprint)
        self.assertNotEqual(one.users[0].position_m, two.users[0].position_m)

    def test_task_bounds_ids_and_all_users_are_valid(self):
        spec = ScenarioSpec()
        scene = generate_scenario(spec, seed=9)
        self.assertEqual(len(scene.tasks), len({task.task_id for task in scene.tasks}))
        self.assertTrue(scene.tasks)
        for task in scene.tasks:
            self.assertIn(task.owner_vehicle_id, range(1, spec.num_users + 1))
            self.assertLess(task.arrival_slot, spec.arrival_horizon_slots)
            self.assertTrue(spec.input_size_bits[0] <= task.input_size_bits <= spec.input_size_bits[1])
            self.assertTrue(spec.output_size_bits[0] <= task.output_size_bits <= spec.output_size_bits[1])
            self.assertTrue(spec.cpu_cycles_per_bit[0] <= task.cpu_cycles_per_bit <= spec.cpu_cycles_per_bit[1])
            self.assertAlmostEqual(task.required_cpu_cycles, task.input_size_bits * task.cpu_cycles_per_bit)
        for server in scene.servers:
            self.assertGreater(server.available_service_cpu_hz, 0.)
            self.assertTrue(scene.mec.covers(server.position_m))
        for vehicle in (*scene.users, *scene.servers):
            expected_heading = 0. if spec.lane_y_m.index(vehicle.position_m[1]) % 2 == 0 else math.pi
            self.assertEqual(vehicle.heading_rad, expected_heading)

    def test_braking_users_move_but_stay_covered_even_through_arbitrarily_long_drain(self):
        scene = generate_scenario(ScenarioSpec(), seed=5)
        for user in scene.users:
            motion = MotionState(user.position_m, user.speed_mps, user.heading_rad, user.acceleration_mps2)
            final = advance_straight_line(motion, 1e5)
            self.assertNotEqual(final.position_m, user.position_m)
            self.assertEqual(final.speed_mps, 0.)
            self.assertTrue(scene.mec.covers(final.position_m))
            expected_travel = user.speed_mps ** 2 / (-2 * user.acceleration_mps2)
            self.assertAlmostEqual(math.dist(user.position_m, final.position_m), expected_travel)

    def test_server_count_and_task_classes_do_not_change_other_rng_streams(self):
        base = ScenarioSpec()
        first = generate_scenario(base, seed=11)
        second = generate_scenario(replace(base, num_servers=1), seed=11)
        self.assertEqual(first.users, second.users)
        self.assertEqual(first.tasks, second.tasks)
        self.assertEqual(first.configuration.max_server_vehicles, second.configuration.max_server_vehicles)
        only_normal = replace(base, task_classes=(TaskClassSpec(Criticality.NORMAL, 1., .6, .1),))
        third = generate_scenario(only_normal, seed=11)
        self.assertEqual([(task.arrival_slot, task.input_size_bits, task.required_cpu_cycles) for task in first.tasks],
                         [(task.arrival_slot, task.input_size_bits, task.required_cpu_cycles) for task in third.tasks])

    def test_envelope_violation_raises_instead_of_discarding_offered_arrivals(self):
        spec = replace(ScenarioSpec(), num_users=1, arrival_horizon_slots=5,
                       max_tasks_per_user=2, arrival_rate_per_user_hz=100.)
        with self.assertRaises(OverflowError):
            generate_scenario(spec, seed=2)
        scene = generate_scenario(replace(spec, max_tasks_per_user=5), seed=2)
        self.assertEqual(len(scene.tasks), 5)
        self.assertEqual(scene.manifest()["arrivals"]["candidate_counts"], [5])
        self.assertEqual(scene.manifest()["arrivals"]["excluded_candidates_by_cap"], [0])

    def test_zero_arrivals_and_zero_servers_are_supported(self):
        scene = generate_scenario(replace(ScenarioSpec(), arrival_rate_per_user_hz=0., num_servers=0), seed=3)
        self.assertEqual(scene.tasks, ())
        self.assertEqual(scene.servers, ())
        environment = scene.environment()
        environment.reset(seed=4)
        while not environment.terminated:
            environment.step({})
        self.assertEqual(environment.metrics()["total_cost"], 0.)

    def test_environments_deep_copy_prototypes_for_fair_comparisons(self):
        scene = generate_scenario(replace(ScenarioSpec(), arrival_horizon_slots=10), seed=4)
        before = scene.fingerprint
        first, second = scene.environment(), scene.environment()
        first.reset(seed=5)
        second.reset(seed=5)
        first.users[1].consume_energy(.1)
        self.assertNotEqual(first.users[1].battery_energy_j, second.users[1].battery_energy_j)
        self.assertEqual(before, scene.fingerprint)

    def test_missing_servers_keep_fixed_neural_action_capacity(self):
        scene = generate_scenario(replace(ScenarioSpec(), num_servers=0,
                                          arrival_horizon_slots=2, arrival_rate_per_user_hz=100.), seed=3)
        environment = scene.environment()
        environment.reset(seed=4)
        while not environment.observations:
            environment.step({})
        for observation in environment.observations.values():
            self.assertEqual(observation.action_mask, (True, False, False, False, False))

    def test_channel_seed_changes_v2v_fading_but_not_scenario_prototypes(self):
        scene = generate_scenario(ScenarioSpec(), seed=12)
        before = scene.fingerprint
        environment = scene.environment()
        environment.reset(seed=1)
        key = (1, scene.servers[0].vehicle_id, "up")
        fading = environment.wireless.links[key].fading_power_gain_linear
        environment.reset(seed=2)
        self.assertNotEqual(fading, environment.wireless.links[key].fading_power_gain_linear)
        self.assertEqual(before, scene.fingerprint)

    def test_specification_json_round_trip_and_invalid_envelopes(self):
        self.assertEqual(ScenarioSpec(), ScenarioSpec.from_dict(json.loads(json.dumps(asdict(ScenarioSpec())))))
        for changes in ({"schema_version": 2}, {"calibration_status": "calibrated"},
                        {"user_braking_magnitude_mps2": (0., 1.)}, {"mec_coverage_radius_m": 50.},
                        {"mec_background_fraction": 1.}, {"server_self_cpu_fraction": (.9, .96)},
                        {"input_size_bits": (4, 1)}, {"user_speed_mps": (float("nan"), 2.)},
                        {"task_classes": (TaskClassSpec(Criticality.NORMAL, .5, 1., 1.),)},
                        {"num_servers": True}, {"arrival_rate_per_user_hz": 101.}):
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                replace(ScenarioSpec(), **changes)
