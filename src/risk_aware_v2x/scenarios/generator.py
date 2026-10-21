"""Explicit independent scenario RNGs and action-independent finite workloads."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum

from ..entities import MECServer, ServerVehicle, Task, UserVehicle
from ..entities._validation import require_non_negative_float, require_positive_float
from ..environment import EnvironmentConfig, SlottedEnvironment
from ..models.channel import LogDistanceChannel
from ..models.randomness import RandomStreams
from .specification import ScenarioSpec


def canonical_json(data) -> str:
    def default(value):
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, set):
            return sorted(value)
        raise TypeError(f"unsupported manifest type {type(value).__name__}")
    return json.dumps(data, sort_keys=True, allow_nan=False, default=default, separators=(",", ":"))


def arrival_probability(rate_hz: float, slot_duration_s: float) -> float:
    """Bernoulli p=lambda*dt, at most one task/user/slot; not Poisson batches."""
    rate = require_non_negative_float("rate_hz", rate_hz)
    duration = require_positive_float("slot_duration_s", slot_duration_s)
    probability = rate * duration
    if not math.isfinite(probability) or probability > 1.:
        raise ValueError("Bernoulli rate*slot duration cannot exceed one")
    return probability


def _road_pose(rng, x_bounds, lanes):
    x, lane_index = rng.uniform(*x_bounds), rng.randrange(len(lanes))
    return (x, lanes[lane_index]), (0. if lane_index % 2 == 0 else math.pi)


@dataclass(frozen=True, slots=True)
class GeneratedScenario:
    specification: ScenarioSpec
    scenario_seed: int
    configuration: EnvironmentConfig
    users: tuple[UserVehicle, ...]
    servers: tuple[ServerVehicle, ...]
    mec: MECServer
    tasks: tuple[Task, ...]
    candidate_counts: tuple[int, ...]

    def environment(self) -> SlottedEnvironment:
        """Each method gets a deep-copied pristine simulator from the same prototypes."""
        return SlottedEnvironment(config=self.configuration, users=self.users,
                                  server_vehicles=self.servers, mec=self.mec, task_trace=self.tasks)

    def manifest(self) -> dict:
        def initial(entity):
            return {key: value for key, value in asdict(entity).items() if not key.startswith("_")}
        p = arrival_probability(self.specification.arrival_rate_per_user_hz,
                                self.specification.slot_duration_s)
        counts = [sum(task.owner_vehicle_id == user.vehicle_id for task in self.tasks) for user in self.users]
        mean_cycles = (sum(self.specification.input_size_bits) / 2
                       * sum(self.specification.cpu_cycles_per_bit) / 2)
        return json.loads(canonical_json({
            "schema_version": 1, "scenario_seed": self.scenario_seed,
            "specification": asdict(self.specification), "configuration": asdict(self.configuration),
            "users": [initial(user) for user in self.users],
            "servers": [initial(server) for server in self.servers], "mec": initial(self.mec),
            "tasks": [initial(task) for task in self.tasks],
            "arrivals": {"probability_per_slot": p,
                         "uncapped_effective_rate_hz": p / self.specification.slot_duration_s,
                         "uncapped_nominal_mec_cpu_load_ratio": (
                             self.specification.num_users * p / self.specification.slot_duration_s * mean_cycles
                             / (self.mec.cpu_capacity_hz - self.mec.background_cpu_load_hz)),
                         "candidate_counts": self.candidate_counts, "admitted_counts": counts,
                         "excluded_candidates_by_cap": [candidate - count for candidate, count in zip(self.candidate_counts, counts, strict=True)]},
        }))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(self.manifest()).encode("utf-8")).hexdigest()


def generate_scenario(spec: ScenarioSpec, *, seed: int) -> GeneratedScenario:
    if not isinstance(spec, ScenarioSpec):
        raise TypeError("spec must be ScenarioSpec")
    streams = RandomStreams(seed)
    v2i = LogDistanceChannel(5.9e9, 2.2)
    config = EnvironmentConfig(
        arrival_horizon_slots=spec.arrival_horizon_slots, max_server_vehicles=spec.max_server_vehicle_slots,
        v2i_uplink=v2i, v2i_downlink=v2i,
        v2v=LogDistanceChannel(5.9e9, spec.v2v_path_loss_exponent,
                               shadowing_std_db=spec.v2v_shadowing_std_db,
                               rayleigh_fading=spec.v2v_rayleigh_fading),
        noise_density_w_per_hz=3.2e-20, v2v_bandwidth_hz=2e6,
        v2v_range_m=spec.v2v_range_m, min_v2v_snr_linear=1., contact_prediction_horizon_s=10.,
        query_request_bits=256, query_header_bits=256, query_record_bits=512,
        query_build_energy_j=1e-6, max_tasks_per_user=spec.max_tasks_per_user,
        max_task_input_bits=spec.input_size_bits[1], max_task_output_bits=spec.output_size_bits[1],
        slot_duration_s=spec.slot_duration_s, information_lifetime_s=spec.information_lifetime_s,
        outage_persistence_slots=2, max_episode_slots=spec.max_episode_slots,
    )
    users, servers, tasks, candidates = [], [], [], []
    for user_id in range(1, spec.num_users + 1):
        rng = streams.stream(f"user-motion:{user_id}")
        position, heading = _road_pose(rng, spec.user_initial_x_m, spec.lane_y_m)
        users.append(UserVehicle(
            vehicle_id=user_id, max_tx_power_w=1., rx_power_w=.5, idle_power_w=.1,
            battery_capacity_j=spec.user_energy_capacity_j, radio_antenna_gain_linear=1.,
            max_task_queue_size=spec.max_tasks_per_user, mobility_profile_id="brake-to-stop-v1",
            position_m=position, speed_mps=rng.uniform(*spec.user_speed_mps), heading_rad=heading,
            acceleration_mps2=-rng.uniform(*spec.user_braking_magnitude_mps2),
        ))
        arrival_rng = streams.stream(f"arrivals:{user_id}")
        p = arrival_probability(spec.arrival_rate_per_user_hz, spec.slot_duration_s)
        slots = [slot for slot in range(spec.arrival_horizon_slots) if arrival_rng.random() < p]
        if len(slots) > spec.max_tasks_per_user:
            raise OverflowError("offered arrivals exceed the configured envelope; increase it, do not discard tasks or reroll seeds")
        candidates.append(len(slots))
        for slot in slots:
            size = streams.stream(f"task-size:{user_id}:{slot}")
            work = streams.stream(f"task-work:{user_id}:{slot}")
            selected = streams.stream(f"task-class:{user_id}:{slot}").choices(
                spec.task_classes, weights=[entry.probability for entry in spec.task_classes], k=1,
            )[0]
            tasks.append(Task(
                task_id=user_id * spec.arrival_horizon_slots + slot, owner_vehicle_id=user_id,
                arrival_slot=slot, input_size_bits=size.randint(*spec.input_size_bits),
                output_size_bits=size.randint(*spec.output_size_bits),
                cpu_cycles_per_bit=work.uniform(*spec.cpu_cycles_per_bit),
                deadline_s=selected.deadline_s, criticality=selected.criticality,
                energy_budget_j=selected.energy_budget_j,
            ))
    for index in range(spec.num_servers):
        server_id = spec.num_users + index + 1
        motion = streams.stream(f"server-motion:{index}")
        resource = streams.stream(f"server-resources:{index}")
        cpu, self_fraction = resource.uniform(*spec.server_cpu_hz), resource.uniform(*spec.server_self_cpu_fraction)
        battery = resource.uniform(*spec.server_battery_j)
        position, heading = _road_pose(motion, spec.server_initial_x_m, spec.lane_y_m)
        servers.append(ServerVehicle(
            vehicle_id=server_id, cpu_capacity_hz=cpu, min_self_reserved_cpu_hz=.05 * cpu,
            max_concurrent_tasks=1, max_queue_size=spec.max_tasks_per_user,
            max_tx_power_w=1., rx_power_w=.5, cpu_energy_coefficient=1e-28,
            battery_capacity_j=battery, min_reserve_energy_j=.1 * battery,
            radio_antenna_gain_linear=1., mobility_profile_id="straight-v1",
            position_m=position, speed_mps=motion.uniform(*spec.server_speed_mps), heading_rad=heading,
            self_workload_cpu_hz=self_fraction * cpu,
        ))
    mec = MECServer(
        server_id=0, position_m=(0., 0.), coverage_radius_m=spec.mec_coverage_radius_m,
        cpu_capacity_hz=spec.mec_cpu_capacity_hz, max_concurrent_tasks=2,
        total_uplink_bandwidth_hz=2e6, total_downlink_bandwidth_hz=2e6,
        max_downlink_power_w=10., query_processing_time_s=.002,
        background_cpu_load_hz=spec.mec_background_fraction * spec.mec_cpu_capacity_hz,
    )
    return GeneratedScenario(spec, seed, config, tuple(users), tuple(servers), mec,
                             tuple(sorted(tasks, key=lambda task: (task.arrival_slot, task.task_id))), tuple(candidates))
