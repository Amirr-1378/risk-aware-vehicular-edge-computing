"""Straight-road kinematics and finite-horizon contact estimates in SI units."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..entities import ServerVehicle, UserVehicle
from ..entities._validation import (
    require_finite_float,
    require_non_negative_float,
    require_position_m,
    require_positive_float,
)


@dataclass(frozen=True, slots=True)
class MotionState:
    """Position in m, speed in m/s, heading in rad, tangential acceleration in m/s²."""

    position_m: tuple[float, float]
    speed_mps: float
    heading_rad: float
    acceleration_mps2: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", require_position_m(self.position_m))
        for name, validate in (
            ("speed_mps", require_non_negative_float),
            ("heading_rad", require_finite_float),
            ("acceleration_mps2", require_finite_float),
        ):
            object.__setattr__(self, name, validate(name, getattr(self, name)))

    @property
    def velocity_mps(self) -> tuple[float, float]:
        """Cartesian velocity in m/s with heading measured counterclockwise from +x."""
        return (
            self.speed_mps * math.cos(self.heading_rad),
            self.speed_mps * math.sin(self.heading_rad),
        )


def distance_m(first_m: tuple[float, float], second_m: tuple[float, float]) -> float:
    """Euclidean separation in metres, including zero separation."""
    return require_non_negative_float(
        "distance_m", math.dist(require_position_m(first_m), require_position_m(second_m))
    )


def advance_straight_line(state: MotionState, duration_s: float) -> MotionState:
    """Integrate constant heading/acceleration, stopping at zero speed under braking.

    The vehicle stays stationary for the rest of the step if it stops before
    the step ends. Road turns and lane changes are controlled by the caller.
    """
    if not isinstance(state, MotionState):
        raise TypeError("state must be a MotionState")
    duration = require_non_negative_float("duration_s", duration_s)
    moving_s = duration
    if state.acceleration_mps2 < 0.0:
        moving_s = min(duration, state.speed_mps / -state.acceleration_mps2)
    travel_m = state.speed_mps * moving_s + 0.5 * state.acceleration_mps2 * moving_s**2
    speed_mps = max(0.0, state.speed_mps + state.acceleration_mps2 * moving_s)
    return MotionState(
        position_m=(
            state.position_m[0] + travel_m * math.cos(state.heading_rad),
            state.position_m[1] + travel_m * math.sin(state.heading_rad),
        ),
        speed_mps=speed_mps,
        heading_rad=state.heading_rad,
        acceleration_mps2=state.acceleration_mps2,
    )


def advance_vehicle(vehicle: UserVehicle | ServerVehicle, duration_s: float) -> MotionState:
    """Validate a mobility step, then install it in the vehicle as one update."""
    if not isinstance(vehicle, (UserVehicle, ServerVehicle)):
        raise TypeError("vehicle must be a UserVehicle or ServerVehicle")
    result = advance_straight_line(
        MotionState(
            vehicle.position_m, vehicle.speed_mps, vehicle.heading_rad,
            vehicle.acceleration_mps2,
        ),
        duration_s,
    )
    vehicle.update_motion(
        position_m=result.position_m, speed_mps=result.speed_mps,
        heading_rad=result.heading_rad, acceleration_mps2=result.acceleration_mps2,
    )
    return result


def estimate_contact_time_s(
    user: MotionState,
    server: MotionState,
    *,
    contact_range_m: float,
    horizon_s: float,
) -> float:
    """Time to first range exit under constant relative velocity, capped at horizon.

    A currently out-of-range pair returns zero, even if it may meet later.
    Acceleration and MEC-region exit are intentionally excluded from this
    predictor. Callers must use cached motion for policy-visible estimates.
    """
    radius = require_positive_float("contact_range_m", contact_range_m)
    horizon = require_non_negative_float("horizon_s", horizon_s)
    # Scaling by range keeps the quadratic in dimensionless relative position.
    relative_p = tuple((b - a) / radius for a, b in zip(user.position_m, server.position_m))
    relative_v = tuple((b - a) / radius for a, b in zip(user.velocity_mps, server.velocity_mps))
    c = math.fsum(value * value for value in relative_p) - 1.0
    if c > 0.0:
        return 0.0
    a = math.fsum(value * value for value in relative_v)
    if a == 0.0:
        return horizon
    b = 2.0 * math.fsum(p * v for p, v in zip(relative_p, relative_v))
    discriminant_root = math.sqrt(max(0.0, b * b - 4.0 * a * c))
    # Stable positive root: avoid cancellation for a nearby outward exit.
    exit_s = (
        -2.0 * c / (b + discriminant_root)
        if b > 0.0
        else (-b + discriminant_root) / (2.0 * a)
    )
    return min(horizon, max(0.0, exit_s))
