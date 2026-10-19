"""Conservative finite-trace user-energy escrow for guaranteed MEC recovery."""

import math
from collections.abc import Iterable

from ..entities import MECServer, Task, UserVehicle
from ..models.channel import noise_power_w, received_sinr_linear, shannon_rate_bps
from ..models.energy import query_energy_j
from ..models.latency import query_response_size_bits
from .config import EnvironmentConfig


class UserEnergyReserves:
    """Reserve a configured envelope, without accessing realized future arrivals.

    Costs consume their own protected budget; V2V can spend only surplus.
    Query completion releases unused query escrow and task completion releases
    unused fallback radio escrow. No battery or transmission rate is fabricated.
    """

    def __init__(self, config: EnvironmentConfig, users: dict[int, UserVehicle],
                 mec: MECServer, task_trace: Iterable[Task], server_count: int) -> None:
        self.users = users
        self.communication_j: dict[int, float] = {}
        self.task_owners: dict[int, int] = {}
        self.unarrived_pool_j: dict[int, float] = {}
        self.unarrived_slots: dict[int, int] = {}
        self.communication_bound_j: dict[int, float] = {}
        self._arrivals_closed = False
        self.query_pool_j: dict[int, float] = {user_id: 0.0 for user_id in users}
        self.query_bound_j: dict[int, float] = {}
        self.query_counts: dict[int, int] = {user_id: 0 for user_id in users}
        self.minimum_uplink_bps: dict[int, float] = {}
        self.minimum_downlink_bps: dict[int, float] = {}
        count = len(users)
        response_bits = query_response_size_bits(config.query_header_bits, config.query_record_bits, server_count)
        for user_id, user in users.items():
            up_band = mec.total_uplink_bandwidth_hz / count
            down_band = mec.total_downlink_bandwidth_hz / count
            up_snr = received_sinr_linear(
                tx_power_w=user.tx_power_w,
                channel_gain_linear=config.v2i_uplink.large_scale_gain_linear(mec.coverage_radius_m),
                noise_w=noise_power_w(config.noise_density_w_per_hz, up_band),
                tx_antenna_gain_linear=user.radio_antenna_gain_linear,
            )
            down_snr = received_sinr_linear(
                tx_power_w=mec.max_downlink_power_w / count,
                channel_gain_linear=config.v2i_downlink.large_scale_gain_linear(mec.coverage_radius_m),
                noise_w=noise_power_w(config.noise_density_w_per_hz, down_band),
                rx_antenna_gain_linear=user.radio_antenna_gain_linear,
            )
            up, down = shannon_rate_bps(up_band, up_snr), shannon_rate_bps(down_band, down_snr)
            if up <= 0 or down <= 0:
                raise ValueError("positive bounded V2I service is required")
            self.minimum_uplink_bps[user_id] = up
            self.minimum_downlink_bps[user_id] = down
            response_slots = math.ceil(response_bits / down / config.slot_duration_s)
            if response_slots * config.slot_duration_s > config.information_lifetime_s:
                raise ValueError("query response can outlive its TTL; increase lifetime or capacity")
            self.query_bound_j[user_id] = query_energy_j(
                build_energy_j=config.query_build_energy_j, tx_power_w=user.tx_power_w,
                request_time_s=config.query_request_bits / up,
                idle_power_w=user.idle_power_w, processing_time_s=mec.query_processing_time_s,
                rx_power_w=user.rx_power_w, response_time_s=response_bits / down,
            )
            self.communication_bound_j[user_id] = (
                user.tx_power_w * config.max_task_input_bits / up
                + user.rx_power_w * config.max_task_output_bits / down
            )
            self.unarrived_slots[user_id] = config.max_tasks_per_user
            self.unarrived_pool_j[user_id] = config.max_tasks_per_user * self.communication_bound_j[user_id]
            self.query_counts[user_id] = config.max_tasks_per_user
            self.query_pool_j[user_id] = config.max_tasks_per_user * self.query_bound_j[user_id]
        # Offline trace validation does not alter the causal runtime escrow.
        counts = {user_id: 0 for user_id in users}
        for task in task_trace:
            counts[task.owner_vehicle_id] += 1
            if (counts[task.owner_vehicle_id] > config.max_tasks_per_user
                    or task.input_size_bits > config.max_task_input_bits
                    or task.output_size_bits > config.max_task_output_bits):
                raise ValueError("task trace exceeds the configured admission/size envelope")
        for user_id, user in users.items():
            if user.battery_energy_j < self.protected_j(user_id):
                raise ValueError(f"user {user_id} lacks the configured MEC/query energy reserve")

    def protected_j(self, user_id: int) -> float:
        return self.query_pool_j[user_id] + self.unarrived_pool_j[user_id] + math.fsum(
            energy for task_id, energy in self.communication_j.items()
            if self.task_owners[task_id] == user_id
        )

    def reveal_task(self, task: Task) -> None:
        """Replace one generic future-task escrow with its now-observed bit counts."""
        user_id, user = task.owner_vehicle_id, self.users[task.owner_vehicle_id]
        if self.unarrived_slots[user_id] <= 0 or self._arrivals_closed:
            raise RuntimeError("arrival exceeded configured envelope")
        self.unarrived_slots[user_id] -= 1
        self.unarrived_pool_j[user_id] = self.unarrived_slots[user_id] * self.communication_bound_j[user_id]
        self.task_owners[task.task_id] = user_id
        self.communication_j[task.task_id] = (
            user.tx_power_w * task.input_size_bits / self.minimum_uplink_bps[user_id]
            + user.rx_power_w * task.output_size_bits / self.minimum_downlink_bps[user_id]
        )

    def close_arrivals(self) -> None:
        """Release unused generic slots only when the configured horizon has ended."""
        if self._arrivals_closed:
            return
        for user_id, count in self.unarrived_slots.items():
            self.query_pool_j[user_id] = max(0.0, self.query_pool_j[user_id] - count * self.query_bound_j[user_id])
            self.query_counts[user_id] -= count
            self.unarrived_pool_j[user_id] = 0.0
        self._arrivals_closed = True

    def start_query(self, user_id: int) -> float:
        if self.query_counts[user_id] <= 0:
            raise RuntimeError("query count exceeded finite-trace admission bound")
        self.query_counts[user_id] -= 1
        return self.query_bound_j[user_id]

    def spend_protected(self, user_id: int, task_id: int, energy_j: float, *, query: bool) -> None:
        remaining = self.query_pool_j[user_id] if query else self.communication_j[task_id]
        if energy_j > remaining and not math.isclose(energy_j, remaining, rel_tol=1e-12, abs_tol=1e-15):
            raise RuntimeError("physical MEC/query energy exceeded the verified escrow bound")
        self.users[user_id].consume_energy(energy_j)
        if query:
            self.query_pool_j[user_id] = max(0.0, remaining - energy_j)
        else:
            self.communication_j[task_id] = max(0.0, remaining - energy_j)

    def finish_query(self, user_id: int, unused_budget_j: float) -> None:
        self.query_pool_j[user_id] = max(0.0, self.query_pool_j[user_id] - unused_budget_j)

    def finish_task(self, task_id: int) -> None:
        self.communication_j.pop(task_id)
