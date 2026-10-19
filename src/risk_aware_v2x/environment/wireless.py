"""Physical per-slot channels, fair spectrum shares, and cached metadata building."""

import math

from ..entities import MECServer, ServerVehicle, Task, UserVehicle
from ..entities.metadata import MECMetadataRecord, MECSnapshot, ServerVehicleMetadata
from ..models.channel import noise_power_w, received_sinr_linear, shannon_rate_bps
from ..models.mobility import MotionState, distance_m, estimate_contact_time_s
from ..models.queueing import queue_workload_cycles
from ..models.randomness import RandomStreams
from .config import EnvironmentConfig


class WirelessState:
    """Owned physical channel state; never returned in a policy observation."""

    def __init__(self, config: EnvironmentConfig, users: dict[int, UserVehicle],
                 servers: dict[int, ServerVehicle], mec: MECServer, streams: RandomStreams) -> None:
        self.config, self.users, self.servers, self.mec, self.streams = config, users, servers, mec, streams
        self.links = {}

    def refresh(self) -> None:
        """One draw per named link at slot boundaries, independent of queries/actions."""
        config, mec = self.config, self.mec
        self.links = {}
        for user_id, user in sorted(self.users.items()):
            for target_id, provider in [(None, mec), *sorted(self.servers.items())]:
                for direction in ("up", "down"):
                    channel = (config.v2i_uplink if direction == "up" else config.v2i_downlink) if target_id is None else config.v2v
                    bandwidth = (mec.total_uplink_bandwidth_hz if direction == "up" else mec.total_downlink_bandwidth_hz) if target_id is None else config.v2v_bandwidth_hz
                    power = user.tx_power_w if direction == "up" else (mec.max_downlink_power_w if target_id is None else provider.max_tx_power_w)
                    name = f"{user_id}:{target_id}:{direction}"
                    self.links[user_id, target_id, direction] = channel.sample(
                        distance_m=distance_m(user.position_m, provider.position_m), tx_power_w=power,
                        bandwidth_hz=bandwidth,
                        noise_w=noise_power_w(config.noise_density_w_per_hz, bandwidth),
                        tx_antenna_gain_linear=user.radio_antenna_gain_linear if direction == "up" else getattr(provider, "radio_antenna_gain_linear", 1.0),
                        rx_antenna_gain_linear=getattr(provider, "radio_antenna_gain_linear", 1.0) if direction == "up" else user.radio_antenna_gain_linear,
                        shadowing_rng=self.streams.stream("shadowing:" + name),
                        fading_rng=self.streams.stream("fading:" + name),
                    )
        mec.registered_server_vehicle_ids = {
            server_id for server_id, server in self.servers.items() if mec.covers(server.position_m)
        }

    def rate_bps(self, user_id: int, server_id: int | None, direction: str, share_count: int = 1) -> float:
        """Instantaneous rate for one equal bandwidth share; MEC DL also shares TX power."""
        config, mec, user = self.config, self.mec, self.users[user_id]
        provider = mec if server_id is None else self.servers[server_id]
        total_bandwidth = (mec.total_uplink_bandwidth_hz if direction == "up" else mec.total_downlink_bandwidth_hz) if server_id is None else config.v2v_bandwidth_hz
        bandwidth = total_bandwidth / share_count
        power = user.tx_power_w if direction == "up" else (mec.max_downlink_power_w / share_count if server_id is None else provider.max_tx_power_w)
        link = self.links[user_id, server_id, direction]
        sinr = received_sinr_linear(
            tx_power_w=power,
            channel_gain_linear=link.large_scale_gain_linear * link.fading_power_gain_linear,
            noise_w=noise_power_w(config.noise_density_w_per_hz, bandwidth),
            tx_antenna_gain_linear=user.radio_antenna_gain_linear if direction == "up" else getattr(provider, "radio_antenna_gain_linear", 1.0),
            rx_antenna_gain_linear=getattr(provider, "radio_antenna_gain_linear", 1.0) if direction == "up" else user.radio_antenna_gain_linear,
        )
        return shannon_rate_bps(bandwidth, sinr)

    def snapshot(self, user_id: int, tasks: dict[int, Task], now_s: float) -> MECSnapshot:
        """Freeze current resource and large-scale link records for one requesting user."""
        user, mec, config = self.users[user_id], self.mec, self.config
        mec_work = queue_workload_cycles(tasks[task_id] for task_id in [*mec.service_queue, *mec.active_task_ids])
        mec_service = mec.cpu_capacity_hz - mec.background_cpu_load_hz
        records = []
        for server_id in sorted(mec.registered_server_vehicle_ids):
            server = self.servers[server_id]
            up, down = self.links[user_id, server_id, "up"], self.links[user_id, server_id, "down"]
            estimated_distance = config.v2v.estimate_distance_m(
                filtered_sinr_linear=up.large_scale_sinr_linear, tx_power_w=user.tx_power_w,
                noise_w=noise_power_w(config.noise_density_w_per_hz, config.v2v_bandwidth_hz),
                tx_antenna_gain_linear=user.radio_antenna_gain_linear,
                rx_antenna_gain_linear=server.radio_antenna_gain_linear,
            )
            user_motion = MotionState(user.position_m, user.speed_mps, user.heading_rad)
            server_motion = MotionState(server.position_m, server.speed_mps, server.heading_rad)
            # Only relative motion and channel-derived distance enter this predictor.
            # Direction is retained from reported positions; magnitude uses the estimate.
            dx, dy = server.position_m[0] - user.position_m[0], server.position_m[1] - user.position_m[1]
            angle = math.atan2(dy, dx)
            estimated_server = MotionState(
                (estimated_distance * math.cos(angle), estimated_distance * math.sin(angle)),
                server.speed_mps, server.heading_rad,
            )
            estimated_user = MotionState((0.0, 0.0), user.speed_mps, user.heading_rad)
            records.append(ServerVehicleMetadata(
                server_id, now_s, max(0.0, server.available_service_cpu_hz),
                queue_workload_cycles(tasks[task_id] for task_id in [*server.service_queue, *server.active_task_ids]),
                server.battery_energy_j / server.battery_capacity_j,
                up.large_scale_sinr_linear, up.large_scale_rate_bps, estimated_distance,
                estimate_contact_time_s(estimated_user, estimated_server, contact_range_m=config.v2v_range_m, horizon_s=config.contact_prediction_horizon_s),
                server.availability,
                server.min_reserve_energy_j / server.battery_capacity_j,
                math.dist(user_motion.velocity_mps, server_motion.velocity_mps),
                True, down.large_scale_rate_bps,
            ))
        return MECSnapshot(now_s, MECMetadataRecord(
            mec.server_id, mec.cpu_capacity_hz, max(0.0, mec.available_cpu_hz), mec_work,
            mec_work / mec_service, max(0.0, mec.available_uplink_bandwidth_hz),
            max(0.0, mec.available_downlink_bandwidth_hz),
            self.links[user_id, None, "up"].large_scale_rate_bps,
            self.links[user_id, None, "down"].large_scale_rate_bps,
        ), tuple(records))
