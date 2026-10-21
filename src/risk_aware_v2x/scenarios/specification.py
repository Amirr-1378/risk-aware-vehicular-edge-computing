"""Versioned SI-valued inputs for a bounded single-cell development workload."""

import math
from dataclasses import dataclass

from ..entities import Criticality
from ..entities._validation import (
    require_finite_float, require_fraction, require_non_negative_float,
    require_positive_float, require_positive_int, require_non_negative_int,
)


@dataclass(frozen=True, slots=True)
class TaskClassSpec:
    criticality: Criticality
    probability: float
    deadline_s: float
    energy_budget_j: float

    def __post_init__(self):
        if not isinstance(self.criticality, Criticality):
            raise TypeError("criticality must be Criticality")
        for name, validator in (("probability", require_fraction),
                                ("deadline_s", require_positive_float),
                                ("energy_budget_j", require_positive_float)):
            object.__setattr__(self, name, validator(name, getattr(self, name)))


def _interval(name, value, validator):
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be an immutable (low, high) pair")
    low, high = (validator(name, entry) for entry in value)
    if low > high:
        raise ValueError(f"{name} has reversed bounds")


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Illustrative straight-road/braking scenario, NOT a calibrated V2X model.

    Range suffixes retain physical units. Within ranges, integers are discrete
    uniform and real values continuous uniform. Lanes are equiprobable; their
    alternating index fixes direction (+x for even, -x for odd lane indices).
    """
    schema_version: int = 1
    label: str = "nominal"
    calibration_status: str = "synthetic-development-only"
    num_users: int = 3
    num_servers: int = 4
    max_server_vehicle_slots: int = 4
    arrival_horizon_slots: int = 200
    max_episode_slots: int = 10_000
    max_tasks_per_user: int = 200
    slot_duration_s: float = .01
    information_lifetime_s: float = .1
    arrival_rate_per_user_hz: float = 3.
    input_size_bits: tuple[int, int] = (32_000, 96_000)
    output_size_bits: tuple[int, int] = (512, 4_096)
    cpu_cycles_per_bit: tuple[float, float] = (300., 700.)
    task_classes: tuple[TaskClassSpec, ...] = (
        TaskClassSpec(Criticality.NORMAL, .7, .6, .1),
        TaskClassSpec(Criticality.HIGH, .2, .3, .1),
        TaskClassSpec(Criticality.CRITICAL, .1, .15, .1),
    )
    lane_y_m: tuple[float, ...] = (-1.75, 1.75)
    user_initial_x_m: tuple[float, float] = (-30., 30.)
    server_initial_x_m: tuple[float, float] = (-100., 100.)
    user_speed_mps: tuple[float, float] = (8., 15.)
    user_braking_magnitude_mps2: tuple[float, float] = (1.5, 2.5)
    server_speed_mps: tuple[float, float] = (8., 25.)
    server_cpu_hz: tuple[float, float] = (3e8, 8e8)
    server_self_cpu_fraction: tuple[float, float] = (.1, .3)
    server_battery_j: tuple[float, float] = (100., 200.)
    user_energy_capacity_j: float = 1_000.
    mec_coverage_radius_m: float = 250.
    mec_cpu_capacity_hz: float = 1.2e9
    mec_background_fraction: float = .1
    v2v_range_m: float = 100.
    v2v_path_loss_exponent: float = 2.7
    v2v_shadowing_std_db: float = 2.
    v2v_rayleigh_fading: bool = True

    def __post_init__(self):
        require_positive_int("schema_version", self.schema_version)
        if self.schema_version != 1 or self.calibration_status != "synthetic-development-only":
            raise ValueError("only uncalibrated development schema v1 is supported")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("a nonempty scenario label is required")
        for name in ("num_users", "arrival_horizon_slots", "max_episode_slots", "max_tasks_per_user"):
            require_positive_int(name, getattr(self, name))
        require_non_negative_int("num_servers", self.num_servers)
        require_non_negative_int("max_server_vehicle_slots", self.max_server_vehicle_slots)
        if self.num_servers > self.max_server_vehicle_slots:
            raise ValueError("server cohort exceeds fixed action capacity")
        if self.max_episode_slots < self.arrival_horizon_slots:
            raise ValueError("guard must include the arrival horizon")
        for name in ("slot_duration_s", "information_lifetime_s", "user_energy_capacity_j",
                     "mec_coverage_radius_m", "mec_cpu_capacity_hz", "v2v_range_m", "v2v_path_loss_exponent"):
            object.__setattr__(self, name, require_positive_float(name, getattr(self, name)))
        for name in ("arrival_rate_per_user_hz", "v2v_shadowing_std_db"):
            object.__setattr__(self, name, require_non_negative_float(name, getattr(self, name)))
        if self.arrival_rate_per_user_hz * self.slot_duration_s > 1.:
            raise ValueError("Bernoulli rate*slot duration cannot exceed one")
        if not isinstance(self.v2v_rayleigh_fading, bool):
            raise TypeError("v2v_rayleigh_fading must be boolean")
        if require_fraction("mec_background_fraction", self.mec_background_fraction) == 1.:
            raise ValueError("MEC must retain service CPU")
        for name in ("input_size_bits", "output_size_bits"):
            _interval(name, getattr(self, name), require_positive_int)
        for name in ("cpu_cycles_per_bit", "user_braking_magnitude_mps2", "server_cpu_hz", "server_battery_j"):
            _interval(name, getattr(self, name), require_positive_float)
        for name in ("user_speed_mps", "server_speed_mps"):
            _interval(name, getattr(self, name), require_non_negative_float)
        for name in ("user_initial_x_m", "server_initial_x_m"):
            _interval(name, getattr(self, name), require_finite_float)
        _interval("server_self_cpu_fraction", self.server_self_cpu_fraction, require_fraction)
        if self.server_self_cpu_fraction[1] >= .95:
            raise ValueError("self-work plus the fixed 5% CPU reservation must leave service CPU")
        if (not isinstance(self.task_classes, tuple) or not self.task_classes
                or not all(isinstance(entry, TaskClassSpec) for entry in self.task_classes)
                or len({entry.criticality for entry in self.task_classes}) != len(self.task_classes)
                or not math.isclose(math.fsum(entry.probability for entry in self.task_classes), 1., abs_tol=1e-12, rel_tol=0)):
            raise ValueError("unique task classes with probabilities summing to one are required")
        if not isinstance(self.lane_y_m, tuple) or not self.lane_y_m:
            raise ValueError("at least one immutable lane coordinate is required")
        lateral = max(abs(require_finite_float("lane_y_m", value)) for value in self.lane_y_m)
        if len(set(self.lane_y_m)) != len(self.lane_y_m):
            raise ValueError("lane coordinates must be distinct")
        # Triangle envelope for every possible braking trajectory, not rejection
        # sampling based on an eventual policy's termination time.
        user_x = max(map(abs, self.user_initial_x_m))
        stop_distance = self.user_speed_mps[1] ** 2 / (2 * self.user_braking_magnitude_mps2[0])
        if math.hypot(user_x + stop_distance, lateral) >= self.mec_coverage_radius_m:
            raise ValueError("user motion envelope can leave the single-MEC region")
        if math.hypot(max(map(abs, self.server_initial_x_m)), lateral) > self.mec_coverage_radius_m:
            raise ValueError("initial server cohort must be inside MEC coverage")

    @classmethod
    def from_dict(cls, data: dict):
        data = dict(data)
        pair_names = ("input_size_bits", "output_size_bits", "cpu_cycles_per_bit", "lane_y_m",
                      "user_initial_x_m", "server_initial_x_m", "user_speed_mps",
                      "user_braking_magnitude_mps2", "server_speed_mps", "server_cpu_hz",
                      "server_self_cpu_fraction", "server_battery_j")
        for name in pair_names:
            if name in data:
                data[name] = tuple(data[name])
        if "task_classes" in data:
            classes = []
            for record in data["task_classes"]:
                record = dict(record)
                record["criticality"] = Criticality(require_positive_int("criticality", record["criticality"]))
                classes.append(TaskClassSpec(**record))
            data["task_classes"] = tuple(classes)
        return cls(**data)
