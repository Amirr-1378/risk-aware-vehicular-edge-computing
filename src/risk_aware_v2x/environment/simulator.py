"""Finite-trace, multi-user slotted offloading environment."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal

from ..entities import (
    ExecutorKind, MECServer, QueryStatus, ServerVehicle, Task, TaskStatus, UserVehicle,
)
from ..entities.metadata import MECSnapshot
from ..entities._validation import require_non_negative_int
from ..models.energy import cpu_energy_j, radio_energy_j
from ..models.latency import WorkProgress, progress_work, query_response_size_bits
from ..models.mobility import advance_straight_line, advance_vehicle, MotionState, distance_m
from ..models.queueing import ordered_tasks, reschedule_resource
from ..models.randomness import RandomStreams
from .accounting import SimulationEvent, TaskLedger, incremental_cost
from .config import EnvironmentConfig
from .observations import OffloadingObservation, build_observation
from .reserves import UserEnergyReserves
from .wireless import WirelessState

_UPLOAD = {TaskStatus.UPLOADING_TO_MEC, TaskStatus.UPLOADING_TO_SERVER_VEHICLE}
_DOWNLOAD = {TaskStatus.DOWNLOADING_FROM_MEC, TaskStatus.DOWNLOADING_FROM_SERVER_VEHICLE}
_COMPUTE = {TaskStatus.MEC_QUEUED, TaskStatus.MEC_EXECUTING, TaskStatus.V2V_QUEUED, TaskStatus.V2V_EXECUTING}
_V2V = {TaskStatus.UPLOADING_TO_SERVER_VEHICLE, TaskStatus.V2V_QUEUED,
        TaskStatus.V2V_EXECUTING, TaskStatus.DOWNLOADING_FROM_SERVER_VEHICLE}


@dataclass(slots=True)
class _Query:
    owner_task_id: int
    request_bits: float
    processing_s: float
    response_bits: float = 0.0
    snapshot: MECSnapshot | None = None
    budget_j: float = 0.0
    spent_j: float = 0.0


@dataclass(frozen=True, slots=True)
class _Operation:
    task_id: int
    phase: str
    progress: WorkProgress
    user_energy_j: float
    server_energy_j: float


@dataclass(frozen=True, slots=True)
class StepResult:
    """One elapsed slot, next decision inputs, costs, and sign-equivalent rewards."""

    slot: int
    observations: dict[int, OffloadingObservation]
    costs: dict[int, float]
    rewards: dict[int, float]
    terminated: bool
    events: tuple[SimulationEvent, ...]


class SlottedEnvironment:
    """Batch decisions before service; physical state is diagnostic, never a policy input.

    Call reset(seed=...), then step({user_id: action}) with exactly the eligible
    users, or {} without decisions. Action 0 is MEC; positive actions resolve
    through that user's received snapshot. All prototypes are deep-copied.
    """

    def __init__(self, *, config: EnvironmentConfig, users: Iterable[UserVehicle],
                 server_vehicles: Iterable[ServerVehicle], mec: MECServer,
                 task_trace: Iterable[Task]) -> None:
        if not isinstance(config, EnvironmentConfig):
            raise TypeError("config must be an EnvironmentConfig")
        users, servers, trace = tuple(users), tuple(server_vehicles), tuple(task_trace)
        if not users or not all(isinstance(user, UserVehicle) for user in users):
            raise ValueError("at least one UserVehicle is required")
        if not all(isinstance(server, ServerVehicle) for server in servers):
            raise TypeError("invalid server vehicle entity")
        if not isinstance(mec, MECServer) or not all(isinstance(task, Task) for task in trace):
            raise TypeError("invalid MEC or task entity")
        for items, attribute in ((users, "vehicle_id"), (servers, "vehicle_id"), (trace, "task_id")):
            ids = [getattr(item, attribute) for item in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {attribute} in scenario")
        if len(servers) > config.max_server_vehicles:
            raise ValueError("scenario exceeds fixed server action capacity")
        if mec.background_cpu_load_hz >= mec.cpu_capacity_hz:
            raise ValueError("MEC must retain positive service CPU for draining")
        for user in users:
            if user.task_queue or user.query_in_flight or user.cached_mec_snapshot is not None:
                raise ValueError("initial users require empty queues and caches")
            if user.tx_power_w <= 0.0 or not mec.covers(user.position_m):
                raise ValueError("users require positive TX power and MEC coverage")
        for provider in (mec, *servers):
            if provider.service_queue or provider.active_task_ids or provider.cpu_allocations_hz:
                raise ValueError("initial compute resources must be empty")
        user_ids = {user.vehicle_id for user in users}
        for task in trace:
            if task.owner_vehicle_id not in user_ids:
                raise ValueError("unknown task owner")
            if task.arrival_slot >= config.arrival_horizon_slots or task.status is not TaskStatus.CREATED:
                raise ValueError("trace tasks must be pristine and arrive before horizon")
        self.config = config
        self._templates = copy.deepcopy((users, servers, mec, trace))
        self._initialized = False

    @property
    def current_time_s(self) -> float:
        return self._slot_time_s(self.slot)

    def _slot_time_s(self, slot: int) -> float:
        return float(Decimal(slot) * Decimal(str(self.config.slot_duration_s)))

    @property
    def observations(self) -> dict[int, OffloadingObservation]:
        return dict(self._observations)

    def reset(self, *, seed: int = 0) -> dict[int, OffloadingObservation]:
        """Restore pristine entities and independent random streams."""
        require_non_negative_int("seed", seed)
        users, servers, self.mec, trace = copy.deepcopy(self._templates)
        self.users = {user.vehicle_id: user for user in sorted(users, key=lambda item: item.vehicle_id)}
        self.server_vehicles = {server.vehicle_id: server for server in sorted(servers, key=lambda item: item.vehicle_id)}
        self._trace = sorted(trace, key=lambda task: (task.arrival_slot, task.owner_vehicle_id, task.task_id))
        self.tasks: dict[int, Task] = {}
        self.ledgers: dict[int, TaskLedger] = {}
        self._next_arrival = 0
        self._queries: dict[int, _Query] = {}
        self._outage_counts: dict[int, int] = {}
        self.events: list[SimulationEvent] = []
        self.slot, self.terminated = 0, False
        self.streams = RandomStreams(seed)
        self.wireless = WirelessState(self.config, self.users, self.server_vehicles, self.mec, self.streams)
        self.reserves = UserEnergyReserves(self.config, self.users, self.mec, self._trace, len(servers))
        self._initialized = True
        self._begin_slot()
        return self.observations

    def _event(self, kind: str, *, task: Task | None = None, user_id: int | None = None,
               time_s: float | None = None, detail: str = "") -> None:
        event = SimulationEvent(kind, self.current_time_s if time_s is None else time_s,
                                task.task_id if task else None,
                                task.owner_vehicle_id if task else user_id, detail)
        self.events.append(event)
        self._slot_events.append(event)

    def _charge(self, task: Task, *, energy_j: float = 0.0, elapsed_s: float = 0.0,
                mec_event: bool = False, deadline_event: bool = False) -> None:
        cost = incremental_cost(task, self.config, energy_j=energy_j, elapsed_s=elapsed_s,
                                mec_event=mec_event, deadline_event=deadline_event)
        self.ledgers[task.task_id].total_cost += cost
        self._slot_costs[task.owner_vehicle_id] += cost

    def _mec_event(self, task: Task, *, time_s: float | None = None) -> None:
        ledger = self.ledgers[task.task_id]
        task.final_executor_kind, task.final_executor_id = ExecutorKind.MEC, self.mec.server_id
        if not ledger.mec_charged:
            ledger.mec_charged = True
            self._charge(task, mec_event=True)
            self._event("mec_selected", task=task, time_s=time_s)

    def _begin_slot(self) -> None:
        self._slot_costs = {user_id: 0.0 for user_id in self.users}
        self._slot_events: list[SimulationEvent] = []
        arrivals, index = [], self._next_arrival
        while index < len(self._trace) and self._trace[index].arrival_slot == self.slot:
            arrivals.append(self._trace[index])
            index += 1
        for user_id, user in self.users.items():
            if len(user.task_queue) + sum(task.owner_vehicle_id == user_id for task in arrivals) > user.max_task_queue_size:
                raise OverflowError("arrival burst exceeds user queue; no task was dropped")
        for task in arrivals:
            self.reserves.reveal_task(task)
            self.tasks[task.task_id], self.ledgers[task.task_id] = task, TaskLedger()
            self.users[task.owner_vehicle_id].enqueue_task(task.task_id)
            self._event("arrival", task=task)
        self._next_arrival = index
        if self.slot >= self.config.arrival_horizon_slots:
            self.reserves.close_arrivals()
        self.wireless.refresh()
        for task in self.tasks.values():
            if task.status is TaskStatus.FALLBACK_PENDING:
                task.prepare_mec_fallback()
                self._mec_event(task)
                self._event("fallback_admitted", task=task)
        self._observations: dict[int, OffloadingObservation] = {}
        for user_id, user in self.users.items():
            self._advance_query_at_boundary(user_id)
            user.refresh_cache_status(self.current_time_s, self.config.information_lifetime_s)
            if not user.task_queue:
                continue
            waiting = ordered_tasks((self.tasks[task_id] for task_id in user.task_queue), self.config.slot_duration_s)
            if not user.query_in_flight and not user.is_cache_valid(self.current_time_s, self.config.information_lifetime_s):
                self._start_query(user_id, waiting[0])
                self._advance_query_at_boundary(user_id)
                user.refresh_cache_status(self.current_time_s, self.config.information_lifetime_s)
            ready = not user.query_in_flight and user.is_cache_valid(self.current_time_s, self.config.information_lifetime_s)
            for task in waiting:
                status = TaskStatus.READY_FOR_OFFLOADING if ready else TaskStatus.WAITING_FOR_INFORMATION
                if task.status is not status:
                    task.transition_to(status)
            if ready:
                self._observations[user_id] = build_observation(user, waiting[0], self.tasks, self.mec, self.config, self.current_time_s)

    def _start_query(self, user_id: int, owner: Task) -> None:
        self.users[user_id].begin_query(owner.task_id, self.current_time_s)
        self._queries[user_id] = _Query(
            owner.task_id, float(self.config.query_request_bits), self.mec.query_processing_time_s,
            budget_j=self.reserves.start_query(user_id),
        )
        self._query_energy(user_id, self.config.query_build_energy_j)
        self._event("query_started", task=owner)

    def _query_energy(self, user_id: int, energy_j: float) -> None:
        query = self._queries[user_id]
        task = self.tasks[query.owner_task_id]
        if (query.spent_j + energy_j > query.budget_j
                and not math.isclose(query.spent_j + energy_j, query.budget_j, rel_tol=1e-12, abs_tol=1e-15)):
            raise RuntimeError("query energy exceeded its individual verified envelope")
        self.reserves.spend_protected(user_id, task.task_id, energy_j, query=True)
        query.spent_j += energy_j
        task.add_query_cost(0.0, energy_j)
        self.ledgers[task.task_id].add_energy("query_j", energy_j)
        self._charge(task, energy_j=energy_j)

    def _advance_query_at_boundary(self, user_id: int) -> None:
        user = self.users[user_id]
        if user_id not in self._queries:
            return
        query = self._queries[user_id]
        if user.query_status is QueryStatus.QUERY_REQUEST_UPLINK and query.request_bits == 0.0:
            user.advance_query_phase(QueryStatus.QUERY_MEC_PROCESSING, query.processing_s)
        if user.query_status is QueryStatus.QUERY_MEC_PROCESSING and query.processing_s == 0.0:
            query.snapshot = self.wireless.snapshot(user_id, self.tasks, self.current_time_s)
            query.response_bits = float(query_response_size_bits(
                self.config.query_header_bits, self.config.query_record_bits, len(query.snapshot.server_vehicles),
            ))
            user.advance_query_phase(QueryStatus.QUERY_RESPONSE_DOWNLINK, 0.0)
            self._event("snapshot_generated", user_id=user_id)
        if user.query_status is QueryStatus.QUERY_RESPONSE_DOWNLINK and query.response_bits == 0.0:
            user.install_snapshot(query.snapshot, self.current_time_s)
            self.reserves.finish_query(user_id, max(0.0, query.budget_j - query.spent_j))
            del self._queries[user_id]
            self._event("snapshot_received", user_id=user_id)
            return
        if user.query_status is QueryStatus.QUERY_REQUEST_UPLINK:
            remaining = query.request_bits / self.wireless.rate_bps(user_id, None, "up")
        elif user.query_status is QueryStatus.QUERY_RESPONSE_DOWNLINK:
            remaining = query.response_bits / self.wireless.rate_bps(user_id, None, "down")
        else:
            remaining = query.processing_s
        user.set_query_remaining_time_s(remaining)

    def step(self, actions: Mapping[int, int]) -> StepResult:
        """Validate a whole batch, then advance one slot. Invalid batches do not mutate state."""
        if not self._initialized:
            raise RuntimeError("call reset before step")
        if self.terminated:
            raise RuntimeError("episode is already terminated")
        if self.slot >= self.config.max_episode_slots:
            raise RuntimeError("episode did not drain within guard; tasks retained")
        if not isinstance(actions, Mapping) or set(actions) != set(self._observations):
            raise ValueError("actions must contain exactly the eligible user IDs")
        for user_id, action in actions.items():
            require_non_negative_int("user_vehicle_id", user_id)
            if isinstance(user_id, bool) or isinstance(action, bool) or not isinstance(action, int):
                raise TypeError("user IDs and actions must be integer identifiers")
            mask = self._observations[user_id].action_mask
            if action < 0 or action >= len(mask) or not mask[action]:
                raise ValueError("action is absent or masked in received observation")
        batch = sorted(actions, key=lambda user_id: (
            *self.tasks[self._observations[user_id].task_id].scheduling_key(self.config.slot_duration_s)[:3],
            user_id, self._observations[user_id].task_id,
        ))
        for user_id in batch:
            observation = self._observations[user_id]
            task, action = self.tasks[observation.task_id], actions[user_id]
            task.assign_target(ExecutorKind.MEC if action == 0 else ExecutorKind.SERVER_VEHICLE,
                               self.mec.server_id if action == 0 else observation.server_id_for_action(action))
            self.users[user_id].remove_task(task.task_id)
            if action == 0:
                self._mec_event(task)
            else:
                self.ledgers[task.task_id].attempt_start_s = self.current_time_s
            self._event("offloaded", task=task, detail="mec" if action == 0 else f"server:{task.initial_target_id}")
        active = {task.task_id: task.status for task in self.tasks.values() if not task.is_terminal}
        self._pending_interruptions: dict[int, str] = {}
        self._blocked: set[int] = set()
        self._check_v2v_links()
        self._schedule_cpu()
        operations, query_radio = self._radio_operations()
        self._progress_queries(query_radio)
        operations.extend(self._cpu_operations())
        busy: dict[int, float] = {}
        uploads: list[int] = []
        computations: list[int] = []
        for operation in sorted(operations, key=lambda entry: self.tasks[entry.task_id].scheduling_key(self.config.slot_duration_s)):
            self._execute_operation(operation, busy, uploads, computations)
        for task_id, status in active.items():
            if status in _COMPUTE:
                active[task_id] = self.tasks[task_id].status
        self._account_elapsed(active, busy)
        for task_id in computations:
            task = self.tasks[task_id]
            resource = self._resource(task)
            resource.release_task(task_id)
            task.transition_to(TaskStatus.DOWNLOADING_FROM_MEC if isinstance(resource, MECServer) else TaskStatus.DOWNLOADING_FROM_SERVER_VEHICLE)
        for task_id in uploads:
            self._queue_uploaded(self.tasks[task_id])
        for task_id, reason in sorted(self._pending_interruptions.items()):
            self._interrupt(self.tasks[task_id], reason)
        for task_id in active:
            task, ledger = self.tasks[task_id], self.ledgers[task_id]
            end_s = task.completion_time_s if task.is_terminal else self._slot_time_s(self.slot + 1)
            if end_s > task.absolute_deadline_s(self.config.slot_duration_s) and not ledger.deadline_charged:
                task.deadline_missed = ledger.deadline_charged = True
                self._charge(task, deadline_event=True)
                self._event("deadline_missed", task=task, time_s=end_s)
        events = tuple(sorted(self._slot_events, key=lambda event: event.time_s))
        if events:
            self.events[-len(events):] = events
        slot, costs = self.slot, dict(self._slot_costs)
        self.slot += 1
        self.terminated = (
            self.slot >= self.config.arrival_horizon_slots and self._next_arrival == len(self._trace)
            and all(task.is_terminal for task in self.tasks.values()) and not self._queries
        )
        if self.terminated:
            self._observations = {}
        else:
            self._advance_mobility()
            self._begin_slot()
        return StepResult(slot, self.observations, costs, {user_id: -cost for user_id, cost in costs.items()}, self.terminated, events)

    def _resource(self, task: Task) -> MECServer | ServerVehicle:
        if task.initial_target_kind is ExecutorKind.MEC or task.fallback_used or task.status is TaskStatus.FALLBACK_PENDING:
            return self.mec
        return self.server_vehicles[task.initial_target_id]

    def _check_v2v_links(self) -> None:
        for task in self.tasks.values():
            if task.status not in _V2V:
                continue
            user, server = self.users[task.owner_vehicle_id], self.server_vehicles[task.initial_target_id]
            reason = ""
            if not self.mec.covers(server.position_m) or not self.mec.covers(user.position_m):
                reason = "mec_region_exit"
            elif distance_m(user.position_m, server.position_m) > self.config.v2v_range_m:
                reason = "contact_range_exit"
            elif not server.availability or server.battery_energy_j <= server.min_reserve_energy_j:
                reason = "server_unavailable"
            elif server.cpu_capacity_hz - server.min_self_reserved_cpu_hz - server.self_workload_cpu_hz <= 0:
                reason = "no_service_cpu"
            direction = "down" if task.status is TaskStatus.DOWNLOADING_FROM_SERVER_VEHICLE else "up"
            snr = self.wireless.links[user.vehicle_id, server.vehicle_id, direction].sinr_linear
            if reason:
                self._pending_interruptions[task.task_id] = reason
                self._blocked.add(task.task_id)
            elif snr < self.config.min_v2v_snr_linear:
                self._blocked.add(task.task_id)
                count = self._outage_counts.get(task.task_id, 0) + 1
                self._outage_counts[task.task_id] = count
                if count >= self.config.outage_persistence_slots:
                    self._pending_interruptions[task.task_id] = "sustained_snr_outage"
            else:
                self._outage_counts[task.task_id] = 0

    def _schedule_cpu(self) -> None:
        for resource in (self.mec, *self.server_vehicles.values()):
            if isinstance(resource, ServerVehicle) and (
                not resource.availability or resource.battery_energy_j <= resource.min_reserve_energy_j
                or resource.cpu_capacity_hz - resource.min_self_reserved_cpu_hz - resource.self_workload_cpu_hz <= 0
            ):
                continue
            reschedule_resource(resource, self.tasks, slot_duration_s=self.config.slot_duration_s)

    def _radio_operations(self) -> tuple[list[_Operation], dict[int, tuple[str, float]]]:
        query_radio: dict[int, tuple[str, float]] = {}
        transfers: list[Task] = []
        occupied_servers: set[int] = set()
        for user_id, user in self.users.items():
            if user.query_in_flight:
                if user.query_status is QueryStatus.QUERY_REQUEST_UPLINK:
                    query_radio[user_id] = ("up", 0.0)
                elif user.query_status is QueryStatus.QUERY_RESPONSE_DOWNLINK:
                    query_radio[user_id] = ("down", 0.0)
                continue
            candidates = ordered_tasks((task for task in self.tasks.values()
                if task.owner_vehicle_id == user_id and task.status in _UPLOAD | _DOWNLOAD
                and task.task_id not in self._blocked), self.config.slot_duration_s)
            if candidates:
                transfers.append(candidates[0])
        selected = []
        for task in ordered_tasks(transfers, self.config.slot_duration_s):
            resource = self._resource(task)
            if isinstance(resource, ServerVehicle):
                if resource.vehicle_id in occupied_servers:
                    continue
                occupied_servers.add(resource.vehicle_id)
            selected.append(task)
        counts = {"up": 0, "down": 0, "v2v": 0}
        for direction, _ in query_radio.values():
            counts[direction] += 1
        for task in selected:
            direction = "up" if task.status in _UPLOAD else "down"
            counts[direction if isinstance(self._resource(task), MECServer) else "v2v"] += 1
        self.mec.uplink_allocations_hz, self.mec.downlink_allocations_hz = {}, {}
        for user_id, (direction, _) in query_radio.items():
            query_radio[user_id] = (direction, self.wireless.rate_bps(user_id, None, direction, counts[direction]))
            allocations = self.mec.uplink_allocations_hz if direction == "up" else self.mec.downlink_allocations_hz
            allocations[user_id] = (self.mec.total_uplink_bandwidth_hz if direction == "up" else self.mec.total_downlink_bandwidth_hz) / counts[direction]
        operations = []
        for task in selected:
            up = task.status in _UPLOAD
            resource, user = self._resource(task), self.users[task.owner_vehicle_id]
            server_id = resource.vehicle_id if isinstance(resource, ServerVehicle) else None
            share = counts["v2v"] if server_id is not None else counts["up" if up else "down"]
            rate = self.wireless.rate_bps(user.vehicle_id, server_id, "up" if up else "down", share)
            progress = progress_work(task.remaining_upload_bits if up else task.remaining_download_bits, rate, self.config.slot_duration_s)
            user_energy = radio_energy_j(user.tx_power_w if up else user.rx_power_w, progress.active_time_s)
            server_energy = radio_energy_j(resource.rx_power_w if up else resource.max_tx_power_w, progress.active_time_s) if server_id is not None else 0.0
            if server_id is None:
                allocations = self.mec.uplink_allocations_hz if up else self.mec.downlink_allocations_hz
                allocations[user.vehicle_id] = (self.mec.total_uplink_bandwidth_hz if up else self.mec.total_downlink_bandwidth_hz) / share
            operations.append(_Operation(task.task_id, "upload" if up else "download", progress, user_energy, server_energy))
        return operations, query_radio

    def _progress_queries(self, query_radio: dict[int, tuple[str, float]]) -> None:
        for user_id, query in self._queries.items():
            user = self.users[user_id]
            if user.query_status is QueryStatus.QUERY_MEC_PROCESSING:
                progress = progress_work(query.processing_s, 1.0, self.config.slot_duration_s)
                query.processing_s = progress.remaining_work
                self._query_energy(user_id, radio_energy_j(user.idle_power_w, progress.active_time_s))
            else:
                direction, rate = query_radio[user_id]
                remaining = query.request_bits if direction == "up" else query.response_bits
                progress = progress_work(remaining, rate, self.config.slot_duration_s)
                if direction == "up":
                    query.request_bits = progress.remaining_work
                else:
                    query.response_bits = progress.remaining_work
                self._query_energy(user_id, radio_energy_j(user.tx_power_w if direction == "up" else user.rx_power_w, progress.active_time_s))

    def _cpu_operations(self) -> list[_Operation]:
        operations = []
        for resource in (self.mec, *self.server_vehicles.values()):
            for task_id, cpu_hz in resource.cpu_allocations_hz.items():
                if task_id in self._blocked:
                    continue
                task = self.tasks[task_id]
                progress = progress_work(task.remaining_cpu_cycles, cpu_hz, self.config.slot_duration_s)
                energy = cpu_energy_j(progress.served_work, cpu_hz, resource.cpu_energy_coefficient) if isinstance(resource, ServerVehicle) else 0.0
                operations.append(_Operation(task_id, "compute", progress, 0.0, energy))
        return operations

    def _execute_operation(self, operation: _Operation, busy: dict[int, float],
                           uploads: list[int], computations: list[int]) -> None:
        task = self.tasks[operation.task_id]
        user, resource = self.users[task.owner_vehicle_id], self._resource(task)
        if isinstance(resource, ServerVehicle):
            reserve = self.reserves.protected_j(user.vehicle_id)
            if not user.can_spend_energy(operation.user_energy_j, reserve):
                self._pending_interruptions[task.task_id] = "user_fallback_reserve"
                return
            if not resource.can_spend_energy(operation.server_energy_j):
                self._pending_interruptions[task.task_id] = "server_energy_reserve"
                return
            user.consume_energy(operation.user_energy_j, reserve)
            resource.consume_energy(operation.server_energy_j)
        else:
            self.reserves.spend_protected(user.vehicle_id, task.task_id, operation.user_energy_j, query=False)
        ledger = self.ledgers[task.task_id]
        busy[task.task_id] = operation.progress.active_time_s
        if operation.user_energy_j:
            ledger.add_energy("user_tx_j" if operation.phase == "upload" else "user_rx_j", operation.user_energy_j)
        if operation.server_energy_j:
            ledger.add_energy({"upload": "server_rx_j", "download": "server_tx_j", "compute": "server_cpu_j"}[operation.phase], operation.server_energy_j)
        self._charge(task, energy_j=operation.user_energy_j + operation.server_energy_j)
        if operation.phase == "upload":
            task.serve_upload(operation.progress.served_work)
            if task.remaining_upload_bits == 0.0:
                uploads.append(task.task_id)
        elif operation.phase == "compute":
            task.serve_cpu(operation.progress.served_work)
            if task.remaining_cpu_cycles == 0.0:
                computations.append(task.task_id)
        else:
            task.serve_download(operation.progress.served_work)
            if task.remaining_download_bits == 0.0:
                task.complete(self.current_time_s + operation.progress.active_time_s, self.config.slot_duration_s,
                              resource.server_id if isinstance(resource, MECServer) else resource.vehicle_id)
                self.reserves.finish_task(task.task_id)
                self._event("completed", task=task, time_s=task.completion_time_s)

    def _account_elapsed(self, starting: dict[int, TaskStatus], busy: dict[int, float]) -> None:
        for task_id, status in starting.items():
            task, ledger = self.tasks[task_id], self.ledgers[task_id]
            elapsed = task.completion_time_s - self.current_time_s if task.is_terminal else self.config.slot_duration_s
            self._charge(task, elapsed_s=elapsed)
            active_s = busy.get(task_id, 0.0)
            if status is TaskStatus.WAITING_FOR_INFORMATION:
                ledger.add_time("query_wait_s", elapsed)
                task.add_query_cost(elapsed, 0.0)
            elif status in {TaskStatus.CREATED, TaskStatus.READY_FOR_OFFLOADING, TaskStatus.FALLBACK_PENDING}:
                ledger.add_time("admission_wait_s", elapsed)
            elif active_s > 0.0:
                bucket = "upload_s" if status in _UPLOAD else "download_s" if status in _DOWNLOAD else "compute_s"
                ledger.add_time(bucket, active_s)
                ledger.add_time("slot_boundary_wait_s", max(0.0, elapsed - active_s))
            else:
                ledger.add_time("queue_wait_s", elapsed)

    def _queue_uploaded(self, task: Task) -> None:
        resource = self._resource(task)
        if isinstance(resource, ServerVehicle) and len(resource.service_queue) >= resource.max_queue_size:
            if len(resource.active_task_ids) < resource.max_concurrent_tasks:
                staged_id = resource.service_queue[0]
                resource.activate_task(staged_id)
                self.tasks[staged_id].transition_to(TaskStatus.V2V_EXECUTING)
            else:
                self._pending_interruptions[task.task_id] = "server_queue_full"
                return
        task.transition_to(TaskStatus.MEC_QUEUED if isinstance(resource, MECServer) else TaskStatus.V2V_QUEUED)
        resource.enqueue_task(task.task_id)

    def _interrupt(self, task: Task, reason: str) -> None:
        server = self.server_vehicles[task.initial_target_id]
        if task.task_id in server.active_task_ids:
            server.release_task(task.task_id)
        elif task.task_id in server.service_queue:
            server.service_queue.remove(task.task_id)
        ledger = self.ledgers[task.task_id]
        elapsed = self._slot_time_s(self.slot + 1) - ledger.attempt_start_s
        spent = math.fsum(getattr(ledger.energy, name) for name in (
            "user_tx_j", "user_rx_j", "user_idle_j", "server_rx_j", "server_cpu_j", "server_tx_j",
        ))
        task.mark_v2v_interrupted(elapsed, spent)
        task.transition_to(TaskStatus.FALLBACK_PENDING)
        ledger.discard_v2v_attempt()
        self._event("v2v_interrupted", task=task, time_s=self._slot_time_s(self.slot + 1), detail=reason)
        self._mec_event(task, time_s=self._slot_time_s(self.slot + 1))

    def _advance_mobility(self) -> None:
        for user in self.users.values():
            motion = advance_straight_line(MotionState(user.position_m, user.speed_mps, user.heading_rad, user.acceleration_mps2), self.config.slot_duration_s)
            if not self.mec.covers(motion.position_m):
                raise ValueError("user leaves single-MEC domain; handover is unsupported")
        for vehicle in (*self.users.values(), *self.server_vehicles.values()):
            advance_vehicle(vehicle, self.config.slot_duration_s)

    def metrics(self) -> dict[str, float]:
        """Diagnostics are separate from policy observations and include unfinished tasks."""
        completed = [task for task in self.tasks.values() if task.is_terminal]
        workload = sum(task.required_cpu_cycles for task in completed)
        return {
            "arrived_tasks": float(len(self.tasks)), "completed_tasks": float(len(completed)),
            "completion_fraction": len(completed) / len(self.tasks) if self.tasks else 0.0,
            "deadline_misses": float(sum(task.deadline_missed for task in self.tasks.values())),
            "v2v_completed_workload_fraction": sum(task.required_cpu_cycles for task in completed if task.final_executor_kind is ExecutorKind.SERVER_VEHICLE) / workload if workload else 0.0,
            "query_count": float(sum(event.kind == "query_started" for event in self.events)),
            "interruption_count": float(sum(event.kind == "v2v_interrupted" for event in self.events)),
            "mobile_energy_j": sum(ledger.energy.total_j for ledger in self.ledgers.values()),
            "total_cost": sum(ledger.total_cost for ledger in self.ledgers.values()),
        }
