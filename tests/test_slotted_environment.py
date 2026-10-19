"""End-to-end fixtures for the environment transition contract and accounting."""

import math
import unittest
from dataclasses import FrozenInstanceError, replace

from risk_aware_v2x.entities import (
    Criticality, ExecutorKind, MECServer, QueryStatus, ServerVehicle, Task,
    TaskStatus, UserVehicle,
)
from risk_aware_v2x.environment import EnvironmentConfig, SlottedEnvironment
from risk_aware_v2x.environment.observations import build_observation
from risk_aware_v2x.models.channel import LogDistanceChannel
from risk_aware_v2x.models.randomness import RandomStreams
from risk_aware_v2x.environment.accounting import incremental_cost


def make_user(vehicle_id: int = 1, **changes: object) -> UserVehicle:
    return UserVehicle(**{
        "vehicle_id": vehicle_id, "max_tx_power_w": 1.0, "rx_power_w": 0.5,
        "idle_power_w": 0.1, "battery_capacity_j": 100.0,
        "radio_antenna_gain_linear": 1.0, "max_task_queue_size": 100,
        "mobility_profile_id": "straight", "position_m": (0.0, 0.0), **changes,
    })


def make_server(vehicle_id: int = 9, **changes: object) -> ServerVehicle:
    return ServerVehicle(**{
        "vehicle_id": vehicle_id, "cpu_capacity_hz": 200_000.0,
        "min_self_reserved_cpu_hz": 10_000.0, "max_concurrent_tasks": 1,
        "max_queue_size": 4, "max_tx_power_w": 1.0, "rx_power_w": 0.5,
        "cpu_energy_coefficient": 1e-20, "battery_capacity_j": 100.0,
        "min_reserve_energy_j": 1.0, "radio_antenna_gain_linear": 1.0,
        "mobility_profile_id": "straight", "position_m": (10.0, 0.0), **changes,
    })


def make_mec(**changes: object) -> MECServer:
    return MECServer(**{
        "server_id": 0, "position_m": (0.0, 0.0), "coverage_radius_m": 500.0,
        "cpu_capacity_hz": 1e6, "max_concurrent_tasks": 1,
        "total_uplink_bandwidth_hz": 1e6, "total_downlink_bandwidth_hz": 1e6,
        "max_downlink_power_w": 10.0, "query_processing_time_s": 0.002, **changes,
    })


def make_task(task_id: int = 1, **changes: object) -> Task:
    return Task(**{
        "task_id": task_id, "owner_vehicle_id": 1, "arrival_slot": 0,
        "input_size_bits": 1_000, "output_size_bits": 100,
        "cpu_cycles_per_bit": 100.0, "deadline_s": 1.0,
        "criticality": Criticality.NORMAL, "energy_budget_j": 1.0, **changes,
    })


def make_config(**changes: object) -> EnvironmentConfig:
    channel = LogDistanceChannel(5.9e9, 2.0)
    return EnvironmentConfig(**{
        "arrival_horizon_slots": 1, "max_server_vehicles": 3,
        "v2i_uplink": channel, "v2i_downlink": channel, "v2v": channel,
        "noise_density_w_per_hz": 4e-21, "v2v_bandwidth_hz": 1e6,
        "v2v_range_m": 100.0, "min_v2v_snr_linear": 1.0,
        "contact_prediction_horizon_s": 10.0, "query_request_bits": 64,
        "query_header_bits": 64, "query_record_bits": 64,
        "query_build_energy_j": 1e-6,
        "max_tasks_per_user": 100, "max_task_input_bits": 4_000,
        "max_task_output_bits": 100, **changes,
    })


def make_env(*, config: EnvironmentConfig | None = None, users=None,
             servers=None, mec=None, tasks=None) -> SlottedEnvironment:
    return SlottedEnvironment(
        config=config or make_config(), users=users if users is not None else [make_user()],
        server_vehicles=servers if servers is not None else [make_server()],
        mec=mec or make_mec(), task_trace=tasks if tasks is not None else [make_task()],
    )


def to_decision(env: SlottedEnvironment, limit: int = 100) -> None:
    for _ in range(limit):
        if env.observations or env.terminated:
            return
        env.step({})
    raise AssertionError("no decision became eligible")


def drain(env: SlottedEnvironment, action: int = 0, limit: int = 1_000):
    results = []
    for _ in range(limit):
        if env.terminated:
            return results
        decisions = {user_id: action if obs.action_mask[action] else 0 for user_id, obs in env.observations.items()}
        results.append(env.step(decisions))
    raise AssertionError("episode did not drain")


class EnvironmentLifecycleTests(unittest.TestCase):
    def test_direct_mec_completes_and_all_component_accounts_reconcile(self) -> None:
        user, server, task = make_user(), make_server(), make_task()
        env = make_env(users=[user], servers=[server], tasks=[task])
        self.assertEqual(env.reset(seed=5), {})
        results = drain(env)
        completed, ledger = env.tasks[1], env.ledgers[1]
        self.assertTrue(completed.is_terminal)
        self.assertEqual(completed.final_executor_kind, ExecutorKind.MEC)
        self.assertAlmostEqual(ledger.latency.total_s, completed.completion_time_s)
        expected = 0.60 * completed.completion_time_s + 0.25 * ledger.energy.total_j + 0.15
        self.assertAlmostEqual(ledger.total_cost, expected)
        self.assertAlmostEqual(sum(result.costs[1] for result in results), expected)
        self.assertAlmostEqual(100.0 - env.users[1].battery_energy_j, ledger.energy.total_j)
        self.assertTrue(all(result.rewards[1] == -result.costs[1] for result in results))
        self.assertEqual(sum(event.kind == "mec_selected" for event in env.events), 1)
        self.assertFalse(env.mec.service_queue or env.mec.active_task_ids)
        self.assertEqual(task.status, TaskStatus.CREATED)
        self.assertEqual(user.battery_energy_j, 100.0)

    def test_v2v_success_counts_mobile_energy_and_no_mec_use(self) -> None:
        env = make_env()
        env.reset(seed=3)
        drain(env, action=1)
        task, ledger = env.tasks[1], env.ledgers[1]
        self.assertEqual(task.final_executor_kind, ExecutorKind.SERVER_VEHICLE)
        self.assertFalse(task.fallback_used or ledger.mec_charged)
        self.assertGreater(ledger.energy.server_cpu_j, 0.0)
        server_energy = ledger.energy.server_rx_j + ledger.energy.server_cpu_j + ledger.energy.server_tx_j
        self.assertAlmostEqual(100.0 - env.server_vehicles[9].battery_energy_j, server_energy)
        self.assertEqual(env.metrics()["v2v_completed_workload_fraction"], 1.0)

    def test_deadline_penalty_occurs_once_while_late_task_continues(self) -> None:
        env = make_env(tasks=[make_task(deadline_s=0.01)])
        env.reset()
        env.step({})
        self.assertFalse(env.tasks[1].deadline_missed)
        env.step({})
        self.assertTrue(env.tasks[1].deadline_missed)
        self.assertFalse(env.tasks[1].is_terminal)
        drain(env)
        self.assertEqual(env.tasks[1].status, TaskStatus.COMPLETED_LATE)
        self.assertEqual(sum(event.kind == "deadline_missed" for event in env.events), 1)
        task, ledger = env.tasks[1], env.ledgers[1]
        expected = 0.60 * task.completion_time_s / task.deadline_s + 0.25 * ledger.energy.total_j + 0.15 + 2.0
        self.assertAlmostEqual(ledger.total_cost, expected)

    def test_exact_deadline_completion_is_on_time(self) -> None:
        reference = make_env(tasks=[make_task(output_size_bits=0)])
        reference.reset()
        drain(reference)
        completion_s = reference.tasks[1].completion_time_s
        env = make_env(tasks=[make_task(output_size_bits=0, deadline_s=completion_s)])
        env.reset()
        drain(env)
        self.assertEqual(env.tasks[1].status, TaskStatus.COMPLETED_ON_TIME)
        self.assertFalse(env.ledgers[1].deadline_charged)

    def test_horizon_ends_before_drain_and_guard_does_not_drop_tasks(self) -> None:
        env = make_env(config=make_config(max_episode_slots=4))
        env.reset()
        for _ in range(4):
            env.step({user_id: 0 for user_id in env.observations})
        self.assertFalse(env.terminated)
        with self.assertRaises(RuntimeError):
            env.step({})
        self.assertFalse(env.tasks[1].is_terminal)

    def test_invalid_batch_is_rejected_before_any_assignment(self) -> None:
        env = make_env(users=[make_user(1), make_user(2)],
                       tasks=[make_task(1), make_task(2, owner_vehicle_id=2)])
        env.reset()
        to_decision(env)
        slot, energy = env.slot, env.users[1].battery_energy_j
        with self.assertRaises(ValueError):
            env.step({1: 0, 2: 3})
        self.assertEqual(env.slot, slot)
        self.assertEqual(env.tasks[1].status, TaskStatus.READY_FOR_OFFLOADING)
        self.assertIsNone(env.tasks[1].initial_target_id)
        self.assertEqual(env.users[1].battery_energy_j, energy)
        with self.assertRaises(ValueError):
            env.step({1: 0})

    def test_reset_repeats_trajectory_and_preserves_input_prototypes(self) -> None:
        env = make_env()
        env.reset(seed=11)
        first = drain(env, 1)
        first_metrics = env.metrics()
        env.reset(seed=11)
        second = drain(env, 1)
        self.assertEqual(first, second)
        self.assertEqual(env.metrics(), first_metrics)


class QueryAndObservationTests(unittest.TestCase):
    def test_task_arriving_mid_query_shares_response_without_repaying_energy(self) -> None:
        env = make_env(config=make_config(arrival_horizon_slots=2),
                       tasks=[make_task(1), make_task(2, arrival_slot=1)])
        env.reset()
        to_decision(env)
        self.assertEqual(env.metrics()["query_count"], 1.0)
        self.assertAlmostEqual(env.tasks[1].query_wait_time_s, .03)
        self.assertAlmostEqual(env.tasks[2].query_wait_time_s, .02)
        self.assertEqual(env.tasks[2].query_energy_j, 0.0)

    def test_long_response_consumes_lifetime_before_reception(self) -> None:
        env = make_env(config=make_config(query_header_bits=500_000))
        env.reset()
        to_decision(env)
        observation = env.observations[1]
        self.assertGreaterEqual(observation.snapshot_age_s, .02)
        self.assertAlmostEqual(observation.remaining_lifetime_s, .1 - observation.snapshot_age_s)

    def test_same_cell_does_not_imply_v2v_range_and_future_arrivals_are_hidden(self) -> None:
        env = make_env(config=make_config(arrival_horizon_slots=21),
                       servers=[make_server(position_m=(200.0, 0.0))],
                       tasks=[make_task(1), make_task(2, arrival_slot=20)])
        env.reset()
        to_decision(env)
        observation = env.observations[1]
        self.assertIsNotNone(observation.servers[0])
        self.assertFalse(observation.action_mask[1])
        self.assertEqual(observation.unassigned_task_count, 1)
        self.assertNotIn(2, env.tasks)

    def test_query_owner_pays_once_but_both_tasks_accumulate_wait(self) -> None:
        env = make_env(tasks=[make_task(1), make_task(2)])
        env.reset()
        to_decision(env)
        self.assertGreater(env.tasks[1].query_energy_j, 0.0)
        self.assertEqual(env.tasks[2].query_energy_j, 0.0)
        self.assertEqual(env.tasks[1].query_wait_time_s, env.tasks[2].query_wait_time_s)
        self.assertEqual(sum(event.kind == "query_started" for event in env.events), 1)
        self.assertEqual(env.users[1].query_status, QueryStatus.CACHE_VALID)

    def test_snapshot_is_aged_at_reception_and_valid_through_exact_ttl_boundary(self) -> None:
        env = make_env()
        env.reset()
        to_decision(env)
        user = env.users[1]
        self.assertLess(user.snapshot_generated_time_s, user.snapshot_received_time_s)
        observation = env.observations[1]
        self.assertGreater(observation.snapshot_age_s, 0.0)
        boundary_s = user.snapshot_generated_time_s + env.config.information_lifetime_s
        self.assertTrue(user.is_cache_valid(boundary_s, env.config.information_lifetime_s))
        self.assertFalse(user.is_cache_valid(math.nextafter(boundary_s, math.inf), env.config.information_lifetime_s))

    def test_cached_observation_and_mask_do_not_read_changed_server_truth(self) -> None:
        env = make_env()
        env.reset()
        to_decision(env)
        before = env.observations[1]
        server = env.server_vehicles[9]
        server.position_m = (400.0, 0.0)
        server.battery_energy_j = 1.0
        server.self_workload_cpu_hz = 100_000.0
        after = build_observation(env.users[1], env.tasks[1], env.tasks, env.mec, env.config, env.current_time_s)
        self.assertEqual(before, after)
        self.assertTrue(after.action_mask[1])
        with self.assertRaises(FrozenInstanceError):
            before.servers[0].record.available_cpu_hz = 0.0

    def test_server_slots_are_sorted_and_absent_slots_are_masked(self) -> None:
        env = make_env(servers=[make_server(12), make_server(9)])
        env.reset()
        to_decision(env)
        observation = env.observations[1]
        self.assertEqual([slot.record.server_vehicle_id if slot else None for slot in observation.servers], [9, 12, None])
        self.assertEqual(observation.action_mask, (True, True, True, False))
        self.assertEqual(observation.schema_version, 1)

    def test_no_registered_server_leaves_mec_as_only_valid_action(self) -> None:
        env = make_env(servers=[make_server(position_m=(600.0, 0.0))])
        env.reset()
        to_decision(env)
        self.assertEqual(env.observations[1].action_mask, (True, False, False, False))
        drain(env)
        self.assertTrue(env.tasks[1].is_terminal)

    def test_expired_idle_cache_does_not_trigger_a_query_until_next_arrival(self) -> None:
        env = make_env(config=make_config(arrival_horizon_slots=40),
                       tasks=[make_task(1), make_task(2, arrival_slot=30)])
        env.reset()
        while env.slot < 29:
            env.step({user_id: 0 for user_id in env.observations})
        self.assertEqual(env.metrics()["query_count"], 1.0)
        env.step({})
        self.assertEqual(env.slot, 30)
        self.assertEqual(env.metrics()["query_count"], 2.0)
        self.assertTrue(env.users[1].query_in_flight)

    def test_queued_ready_tasks_wait_again_when_a_short_ttl_expires(self) -> None:
        env = make_env(config=make_config(information_lifetime_s=0.01),
                       tasks=[make_task(1), make_task(2), make_task(3)])
        env.reset()
        to_decision(env)
        env.step({1: 0})
        self.assertEqual(env.tasks[2].status, TaskStatus.WAITING_FOR_INFORMATION)
        drain(env)
        self.assertEqual(env.metrics()["completed_tasks"], 3.0)
        self.assertLessEqual(env.metrics()["query_count"], 3.0)


class RecoveryAndBatchTests(unittest.TestCase):
    def test_stale_valid_action_leads_to_full_mec_restart_and_one_use_charge(self) -> None:
        env = make_env()
        env.reset()
        to_decision(env)
        self.assertTrue(env.observations[1].action_mask[1])
        env.server_vehicles[9].position_m = (400.0, 0.0)
        env.step({1: 1})
        task = env.tasks[1]
        self.assertEqual(task.status, TaskStatus.UPLOADING_TO_MEC)
        self.assertTrue(task.fallback_used)
        self.assertEqual(task.remaining_upload_bits, task.input_size_bits)
        self.assertEqual(task.remaining_cpu_cycles, task.required_cpu_cycles)
        self.assertEqual(task.remaining_download_bits, task.output_size_bits)
        drain(env)
        self.assertEqual(task.final_executor_kind, ExecutorKind.MEC)
        self.assertEqual(sum(event.kind == "mec_selected" for event in env.events), 1)
        self.assertEqual(env.metrics()["completion_fraction"], 1.0)

    def test_interruption_after_partial_compute_preserves_wasted_work_cost(self) -> None:
        env = make_env()
        env.reset()
        to_decision(env)
        env.step({1: 1})
        env.step({})
        task = env.tasks[1]
        self.assertLess(task.remaining_cpu_cycles, task.required_cpu_cycles)
        before_j = env.ledgers[1].energy.total_j
        query_j = task.query_energy_j
        env.server_vehicles[9].position_m = (600.0, 0.0)
        env.step({})
        self.assertEqual(task.remaining_cpu_cycles, task.required_cpu_cycles)
        self.assertAlmostEqual(env.ledgers[1].energy.total_j, before_j)
        self.assertAlmostEqual(task.v2v_attempt_energy_j, before_j - query_j)
        drain(env)
        ledger = env.ledgers[1]
        self.assertAlmostEqual(ledger.latency.total_s, task.completion_time_s)
        self.assertGreater(ledger.energy.interrupted_attempt_j, 0.0)
        expected = .60 * task.completion_time_s + .25 * ledger.energy.total_j + .15
        self.assertAlmostEqual(ledger.total_cost, expected)
        self.assertFalse(env.server_vehicles[9].service_queue or env.server_vehicles[9].active_task_ids)

    def test_server_energy_reserve_interrupts_after_partial_processing(self) -> None:
        env = make_env(servers=[make_server(cpu_energy_coefficient=1e-14, battery_energy_j=1.8)])
        env.reset()
        to_decision(env)
        env.step({1: 1})
        env.step({})
        self.assertLess(env.tasks[1].remaining_cpu_cycles, env.tasks[1].required_cpu_cycles)
        env.step({})
        self.assertTrue(env.tasks[1].fallback_used)
        self.assertGreaterEqual(env.server_vehicles[9].battery_energy_j, 1.0)
        self.assertIn("server_energy_reserve", [event.detail for event in env.events if event.kind == "v2v_interrupted"])
        drain(env)

    def test_user_backup_energy_cannot_be_spent_on_v2v(self) -> None:
        config = make_config(query_request_bits=0, query_header_bits=0,
                             query_record_bits=0, query_build_energy_j=0.0,
                             max_tasks_per_user=1, max_task_input_bits=1_000)
        reference = make_env(config=config, mec=make_mec(query_processing_time_s=0.0))
        reference.reset()
        budget = reference.reserves.protected_j(1)
        env = make_env(config=config, users=[make_user(battery_energy_j=budget)],
                       mec=make_mec(query_processing_time_s=0.0))
        env.reset()
        self.assertTrue(env.observations[1].action_mask[1])
        env.step({1: 1})
        self.assertTrue(env.tasks[1].fallback_used)
        self.assertIn("user_fallback_reserve", [event.detail for event in env.events if event.kind == "v2v_interrupted"])
        drain(env)
        self.assertTrue(env.tasks[1].is_terminal)
        self.assertGreaterEqual(env.users[1].battery_energy_j, 0.0)

    def test_outage_persistence_stalls_then_recovers_through_mec(self) -> None:
        env = make_env(config=make_config(outage_persistence_slots=2))
        env.reset()
        to_decision(env)
        for index in range(2):
            link = env.wireless.links[1, 9, "up"]
            env.wireless.links[1, 9, "up"] = replace(link, sinr_linear=0.0, fading_power_gain_linear=0.0)
            env.step({1: 1} if index == 0 else {})
            self.assertEqual(env.tasks[1].fallback_used, index == 1)
        self.assertEqual(env.tasks[1].remaining_upload_bits, 1_000.0)
        drain(env)
        self.assertTrue(env.tasks[1].is_terminal)

    def test_mec_critical_arrival_preempts_at_next_boundary_without_losing_work(self) -> None:
        env = make_env(config=make_config(arrival_horizon_slots=7), tasks=[
            make_task(1), make_task(2, arrival_slot=6, criticality=Criticality.CRITICAL,
                                    input_size_bits=10, cpu_cycles_per_bit=1.0),
        ])
        env.reset()
        while env.slot < 7:
            env.step({user_id: 0 for user_id in env.observations})
        normal = env.tasks[1]
        remaining = normal.remaining_cpu_cycles
        env.step({})
        self.assertEqual(normal.status, TaskStatus.MEC_QUEUED)
        self.assertEqual(normal.remaining_cpu_cycles, remaining)
        self.assertEqual(env.tasks[2].status, TaskStatus.DOWNLOADING_FROM_MEC)
        drain(env)
        self.assertEqual(env.metrics()["completed_tasks"], 2.0)

    def test_full_server_queue_routes_overflow_to_mec_and_drains_every_task(self) -> None:
        env = make_env(users=[make_user(1), make_user(2), make_user(3)],
                       servers=[make_server(max_queue_size=1)], tasks=[
                           make_task(1), make_task(2, owner_vehicle_id=2), make_task(3, owner_vehicle_id=3),
                       ])
        env.reset()
        drain(env, action=1)
        self.assertEqual(env.metrics()["completed_tasks"], 3.0)
        self.assertIn("server_queue_full", [event.detail for event in env.events if event.kind == "v2v_interrupted"])
        self.assertTrue(all(task.is_terminal for task in env.tasks.values()))

    def test_user_and_action_mapping_order_does_not_change_batch_trajectory(self) -> None:
        def run(reverse):
            env = make_env(users=[make_user(2), make_user(1)] if reverse else [make_user(1), make_user(2)],
                           tasks=[make_task(2, owner_vehicle_id=2), make_task(1)])
            env.reset(seed=9)
            results = []
            while not env.terminated:
                ids = sorted(env.observations, reverse=reverse)
                results.append(env.step({user_id: 0 for user_id in ids}))
            return env.metrics(), results
        self.assertEqual(run(False), run(True))


class ScenarioValidationTests(unittest.TestCase):
    def test_unknown_owners_duplicate_ids_and_out_of_horizon_arrivals(self) -> None:
        for tasks in ([make_task(owner_vehicle_id=5)], [make_task(), make_task()], [make_task(arrival_slot=1)]):
            with self.subTest(tasks=tasks):
                with self.assertRaises(ValueError):
                    make_env(tasks=tasks)

    def test_unfunded_energy_or_expiring_response_is_rejected_at_reset(self) -> None:
        with self.assertRaises(ValueError):
            make_env(users=[make_user(battery_energy_j=0.0)]).reset()
        with self.assertRaises(ValueError):
            make_env(config=make_config(information_lifetime_s=0.001)).reset()

    def test_stochastic_v2i_and_zero_mec_cpu_are_outside_guarantee(self) -> None:
        with self.assertRaises(ValueError):
            make_config(v2i_uplink=LogDistanceChannel(5.9e9, 2.0, rayleigh_fading=True))
        with self.assertRaises(ValueError):
            make_env(mec=make_mec(background_cpu_load_hz=1e6))

    def test_user_queue_overflow_is_an_explicit_error_and_not_silent_loss(self) -> None:
        env = make_env(users=[make_user(max_task_queue_size=1)], tasks=[make_task(1), make_task(2)])
        with self.assertRaises(OverflowError):
            env.reset()
        self.assertEqual(env.tasks, {})

    def test_user_leaving_mec_domain_is_reported_instead_of_faking_connectivity(self) -> None:
        env = make_env(users=[make_user(position_m=(499.0, 0.0), speed_mps=200.0)])
        env.reset()
        with self.assertRaises(ValueError):
            env.step({})


class EnvironmentScientificTests(unittest.TestCase):
    def test_future_trace_changes_do_not_change_present_escrow_or_observations(self) -> None:
        config = make_config(arrival_horizon_slots=21)
        first = make_env(config=config, tasks=[make_task(1), make_task(2, arrival_slot=20)])
        second = make_env(config=config, tasks=[make_task(1),
            make_task(2, arrival_slot=20, input_size_bits=4_000),
            make_task(3, arrival_slot=20, input_size_bits=2_000),
        ])
        first.reset(seed=7)
        second.reset(seed=7)
        self.assertEqual(first.reserves.protected_j(1), second.reserves.protected_j(1))
        for _ in range(10):
            a = first.step({user_id: 0 for user_id in first.observations})
            b = second.step({user_id: 0 for user_id in second.observations})
            self.assertEqual(a, b)
            self.assertEqual(first.reserves.protected_j(1), second.reserves.protected_j(1))

    def test_combined_cost_fixture_and_criticality_scaling(self) -> None:
        # 0.6*(0.5/1) + 0.25*(0.2/1) + 0.15 + 2 = 2.5.
        for criticality, expected in ((Criticality.NORMAL, 2.5), (Criticality.HIGH, 5.0), (Criticality.CRITICAL, 10.0)):
            with self.subTest(criticality=criticality):
                value = incremental_cost(make_task(criticality=criticality), make_config(),
                                         elapsed_s=.5, energy_j=.2, mec_event=True, deadline_event=True)
                self.assertAlmostEqual(value, expected)
        with self.assertRaises(ValueError):
            incremental_cost(make_task(), make_config(), elapsed_s=-1.0)

    def test_additional_fading_draws_leave_arrivals_and_mobility_unchanged(self) -> None:
        channel = LogDistanceChannel(5.9e9, 2.0, rayleigh_fading=True)
        config = make_config(v2v=channel)
        first, second = make_env(config=config), make_env(config=config)
        first.reset(seed=10)
        second.reset(seed=10)
        for _ in range(50):
            first.streams.stream("fading:1:9:up").random()
        first_result = first.step({})
        second_result = second.step({})
        self.assertNotEqual(first.wireless.links[1, 9, "up"].fading_power_gain_linear,
                            second.wireless.links[1, 9, "up"].fading_power_gain_linear)
        self.assertEqual(first_result, second_result)
        self.assertEqual(first.users[1].position_m, second.users[1].position_m)
        self.assertEqual(first.server_vehicles[9].position_m, second.server_vehicles[9].position_m)
        self.assertEqual(list(first.tasks), list(second.tasks))

    def test_seeded_multi_user_episode_conserves_all_tasks_energy_and_cost(self) -> None:
        streams = RandomStreams(123)
        arrivals, policy = streams.stream("arrivals"), streams.stream("agent")
        trace = [make_task(
            task_id + 1, owner_vehicle_id=task_id % 3 + 1,
            arrival_slot=arrivals.randrange(50), input_size_bits=arrivals.randrange(1_000, 4_000),
            criticality=arrivals.choice(list(Criticality)), deadline_s=.5,
        ) for task_id in range(24)]
        config = make_config(arrival_horizon_slots=60,
                             v2v=LogDistanceChannel(5.9e9, 2.0, shadowing_std_db=3.0, rayleigh_fading=True))
        env = make_env(config=config, users=[make_user(user_id) for user_id in (1, 2, 3)],
                       servers=[make_server(9), make_server(10, position_m=(50.0, 0.0))], tasks=trace)
        env.reset(seed=99)
        returned_cost = 0.0
        for _ in range(5_000):
            if env.terminated:
                break
            actions = {user_id: policy.choice([index for index, enabled in enumerate(obs.action_mask) if enabled])
                       for user_id, obs in env.observations.items()}
            result = env.step(actions)
            returned_cost += sum(result.costs.values())
        self.assertTrue(env.terminated)
        self.assertEqual(env.metrics()["completed_tasks"], 24.0)
        self.assertAlmostEqual(returned_cost, env.metrics()["total_cost"])
        physical_energy = sum(100.0 - user.battery_energy_j for user in env.users.values())
        physical_energy += sum(100.0 - server.battery_energy_j for server in env.server_vehicles.values())
        self.assertAlmostEqual(physical_energy, env.metrics()["mobile_energy_j"])
        for task_id, task in env.tasks.items():
            ledger = env.ledgers[task_id]
            elapsed_s = task.completion_time_s - task.arrival_slot * config.slot_duration_s
            self.assertAlmostEqual(ledger.latency.total_s, elapsed_s)
            expected = int(task.criticality) * (
                .60 * elapsed_s / task.deadline_s + .25 * ledger.energy.total_j / task.energy_budget_j
                + .15 * (task.final_executor_kind is ExecutorKind.MEC) + 2.0 * task.deadline_missed
            )
            self.assertAlmostEqual(ledger.total_cost, expected)
