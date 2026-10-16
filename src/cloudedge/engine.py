"""Shared SimPy event kernel for actual execution and candidate replay.

Resource queues are explicit serializable state instead of generator stacks.
Completion/propagation events have priority 0; deadlines priority 2. This also
handles multiple zero-duration transitions at the exact deadline correctly.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import simpy

from .models import EventRecord, ResourceState, RuntimeTask, Scenario, Score, Snapshot, StageTiming, Task


PHASES = ("wlan_upload", "wan_upload", "compute", "wan_download", "wlan_download")


def route(task: Task, vm: int, scenario: Scenario) -> tuple[str, ...]:
    if scenario.vms[vm].tier == "cloud":
        return PHASES
    return ("wlan_upload", "compute", "wlan_download")


def resource_for(task: Task, vm: int, phase: str) -> str:
    if phase == "compute":
        return f"vm_{vm}"
    if phase.startswith("wlan"):
        return f"wlan_{task.ap}_{'up' if phase.endswith('upload') else 'down'}"
    return f"wan_{'up' if phase.endswith('upload') else 'down'}"


def service_time(task: Task, vm: int, phase: str, scenario: Scenario) -> float:
    if phase == "compute":
        return task.mi / scenario.vms[vm].mips
    size = task.input_bytes if phase.endswith("upload") else task.output_bytes
    resource = resource_for(task, vm, phase)
    link = next(link for link in scenario.links if link.id == resource)
    return size * 8 / (link.mbps * 1_000_000)


def propagation_time(task: Task, vm: int, phase: str, scenario: Scenario) -> float:
    if phase == "compute":
        return 0.
    resource = resource_for(task, vm, phase)
    return next(link.propagation for link in scenario.links if link.id == resource)


class Engine:
    def __init__(self, scenario: Scenario, now: float = 0.):
        self.scenario = scenario
        self.env = simpy.Environment(initial_time=now)
        self.tasks: dict[int, RuntimeTask] = {}
        self.live: set[int] = set()
        self.pending: set[int] = set()
        self.resources = {f"vm_{vm.id}": ResourceState(f"vm_{vm.id}", vm.slots) for vm in scenario.vms}
        self.resources.update({link.id: ResourceState(link.id) for link in scenario.links})
        self.events: dict[int, EventRecord] = {}
        self.next_id = 0

    @property
    def now(self) -> float:
        return float(self.env.now)

    def _bind(self, record: EventRecord) -> None:
        event = simpy.Event(self.env)
        event._ok = True
        event._value = None
        event.callbacks.append(lambda _: self._handle(record.id))
        self.env.schedule(event, priority=record.priority, delay=max(0., record.time - self.now))

    def _schedule(self, kind: str, time: float, task_id: int, data=None, priority: int = 0) -> int:
        time = round(float(time), 12)
        if time < self.now - 1e-10:
            raise ValueError("Cannot schedule an event in the past")
        record = EventRecord(self.next_id, max(time, self.now), priority, kind, task_id, data)
        self.next_id += 1
        self.events[record.id] = record
        if task_id in self.tasks:
            self.tasks[task_id].events.add(record.id)
        self._bind(record)
        return record.id

    def advance(self, time: float) -> None:
        time = round(float(time), 12)
        if time < self.now - 1e-10:
            raise ValueError("Simulation time cannot move backwards")
        # Unlike env.run(until=numeric), this processes events AT the boundary.
        while self.env.peek() <= time:
            self.env.step()
        if self.now < time:
            event = simpy.Event(self.env)
            event._ok, event._value = True, None
            self.env.schedule(event, priority=3, delay=time - self.now)
            while self.env.peek() <= time:
                self.env.step()

    def add_task(self, task: Task) -> None:
        if task.id in self.tasks:
            raise ValueError(f"Duplicate task ID {task.id}")
        if task.release > self.now + 1e-10 or task.deadline < self.now - 1e-10:
            raise ValueError("Task must have arrived and not be expired")
        self.tasks[task.id] = RuntimeTask(task)
        self.live.add(task.id)
        self.pending.add(task.id)
        self._schedule("deadline", task.deadline, task.id, priority=2)

    def release_assigned(self, task: Task, vm: int) -> None:
        if task.release <= self.now:
            self.add_task(task)
            self.dispatch(task.id, vm)
        else:
            self._schedule("release", task.release, task.id, (task, vm))

    def dispatch(self, task_id: int, vm: int) -> None:
        rt = self.tasks[task_id]
        if rt.status != "pending" or not isinstance(vm, (int, np.integer)) or not 0 <= int(vm) < len(self.scenario.vms):
            raise ValueError("Dispatch requires a pending task and valid VM")
        rt.vm, rt.dispatched, rt.status = int(vm), self.now, "running"
        self.pending.remove(task_id)
        self._enter_phase(rt)

    def _enter_phase(self, rt: RuntimeTask) -> None:
        path = route(rt.task, rt.vm, self.scenario)
        if rt.phase == len(path):
            self._finish(rt, "success")
            return
        phase = path[rt.phase]
        rt.stage = f"{phase}_queue"
        rt.timings[phase] = StageTiming(self.now)
        resource = self.resources[resource_for(rt.task, rt.vm, phase)]
        resource.queue.append(rt.task.id)
        self._pump(resource)

    def _pump(self, resource: ResourceState) -> None:
        while resource.queue and len(resource.active) < resource.capacity:
            task_id = resource.queue.pop(0)
            rt = self.tasks[task_id]
            if rt.finished is not None:
                continue
            phase = route(rt.task, rt.vm, self.scenario)[rt.phase]
            rt.stage = phase
            rt.timings[phase].started = self.now
            end = round(self.now + service_time(rt.task, rt.vm, phase, self.scenario), 12)
            resource.active[task_id] = (self.now, end)
            self._schedule("service_done", end, task_id, resource.id)

    def _finish(self, rt: RuntimeTask, status: str) -> None:
        rt.status, rt.finished = status, self.now
        self.live.discard(rt.task.id)
        self.pending.discard(rt.task.id)
        if status == "failed":
            rt.failure_reason = rt.stage
        for event_id in list(rt.events):
            self.events.pop(event_id, None)
        rt.events.clear()
        for resource in self.resources.values():
            if rt.task.id in resource.queue:
                resource.queue.remove(rt.task.id)
            if rt.task.id in resource.active:
                start, _ = resource.active.pop(rt.task.id)
                resource.intervals.append((start, self.now))
                self._pump(resource)

    def _handle(self, event_id: int) -> None:
        record = self.events.pop(event_id, None)
        if record is None:  # canceled event still in SimPy's heap
            return
        if record.kind == "release":
            task, vm = record.data
            self.add_task(task)
            self.dispatch(task.id, vm)
            return
        rt = self.tasks[record.task_id]
        rt.events.discard(event_id)
        if rt.finished is not None:
            return
        if record.kind == "deadline":
            self._finish(rt, "failed")
        elif record.kind == "service_done":
            resource = self.resources[record.data]
            start, _ = resource.active.pop(rt.task.id)
            resource.intervals.append((start, self.now))
            phase = route(rt.task, rt.vm, self.scenario)[rt.phase]
            rt.timings[phase].service_end = self.now
            rt.stage = f"{phase}_propagation"
            self._pump(resource)
            self._schedule("advance", self.now + propagation_time(rt.task, rt.vm, phase, self.scenario), rt.task.id)
        elif record.kind == "advance":
            phase = route(rt.task, rt.vm, self.scenario)[rt.phase]
            rt.timings[phase].finished = self.now
            rt.phase += 1
            self._enter_phase(rt)
        else:
            raise RuntimeError(f"Unknown event {record.kind}")

    def snapshot(self) -> Snapshot:
        live = {key: deepcopy(self.tasks[key]) for key in sorted(self.live)}
        resources = {
            key: ResourceState(key, r.capacity, list(r.queue), dict(r.active))
            for key, r in self.resources.items()
        }
        events = tuple(deepcopy(e) for e in self.events.values() if e.task_id in live)
        return Snapshot(self.now, self.scenario, live, resources, events, self.next_id)

    @classmethod
    def from_snapshot(cls, snapshot: Snapshot) -> Engine:
        engine = cls(snapshot.scenario, snapshot.now)
        engine.tasks = deepcopy(snapshot.tasks)
        engine.live = set(engine.tasks)
        engine.pending = {key for key, rt in engine.tasks.items() if rt.status == "pending"}
        engine.resources = deepcopy(snapshot.resources)
        engine.next_id = snapshot.next_id
        for record in sorted(deepcopy(snapshot.events), key=lambda e: (e.time, e.priority, e.id)):
            engine.events[record.id] = record
            engine._bind(record)
        return engine

    def utilization(self, resource: str, start: float, end: float) -> float:
        if end <= start:
            return 0.
        state = self.resources[resource]
        intervals = state.intervals + [(s, min(self.now, e)) for s, e in state.active.values()]
        busy = sum(max(0., min(end, b) - max(start, a)) for a, b in intervals)
        return min(1., busy / ((end - start) * state.capacity))

    def score(self, tasks: list[Task], start: float, end: float) -> Score:
        if not tasks:
            return Score(0., 0., 0., 0.)
        outcomes = [self.tasks[t.id] for t in tasks]
        if any(rt.finished is None for rt in outcomes):
            raise RuntimeError("Cannot score unfinished tasks")
        failure = sum(rt.status == "failed" for rt in outcomes) / len(outcomes)
        delay = sum(1. if rt.status == "failed" else min(1., (rt.finished - rt.task.release) / rt.task.max_delay) for rt in outcomes) / len(outcomes)
        edge_loads = [self.utilization(f"vm_{v.id}", start, end) for v in self.scenario.vms if v.tier == "edge"]
        imbalance = min(1., 2 * float(np.std(edge_loads)))
        weights = self.scenario.weights
        return Score(failure, delay, imbalance, weights[0] * failure + weights[1] * delay + weights[2] * imbalance)

    def task_rows(self) -> list[dict]:
        rows = []
        for rt in sorted(self.tasks.values(), key=lambda r: r.task.id):
            if rt.finished is None:
                raise RuntimeError("Run contains unfinished tasks")
            task = rt.task
            row = dict(task_id=task.id, device=task.device, ap=task.ap, app=task.app, release=task.release,
                deadline=task.deadline, max_delay=task.max_delay, mi=task.mi, input_bytes=task.input_bytes,
                output_bytes=task.output_bytes, vm=rt.vm,
                tier=None if rt.vm is None else self.scenario.vms[rt.vm].tier,
                dispatched=rt.dispatched, finished=rt.finished, status=rt.status,
                failure_reason=rt.failure_reason, latency_s=rt.finished-task.release,
                scheduler_wait_s=(rt.finished if rt.dispatched is None else rt.dispatched)-task.release)
            for phase in PHASES:
                timing = rt.timings.get(phase)
                waiting, service, propagation = timing.durations(rt.finished) if timing else (0., 0., 0.)
                row.update({f"{phase}_queue_s": waiting, f"{phase}_service_s": service, f"{phase}_propagation_s": propagation})
                for attr in ("requested", "started", "service_end", "finished"):
                    row[f"{phase}_{attr}"] = getattr(timing, attr) if timing else None
            rows.append(row)
        return rows


class Evaluator:
    def evaluate(self, assignments: dict[int, int], batch: list[Task], snapshot: Snapshot) -> Score:
        if set(assignments) != {t.id for t in batch}:
            raise ValueError("Assignments must cover exactly the current batch")
        if not batch:
            return Score(0., 0., 0., 0.)
        engine = Engine.from_snapshot(snapshot)
        for task in batch:
            if task.id in engine.tasks:
                if engine.tasks[task.id].task != task:
                    raise ValueError("Batch task differs from snapshot")
                engine.dispatch(task.id, assignments[task.id])
            else:
                if snapshot.scenario.mode == "dynamic" and task.release > snapshot.now:
                    raise ValueError("Dynamic evaluation cannot see future arrivals")
                if not isinstance(assignments[task.id], (int, np.integer)) or not 0 <= int(assignments[task.id]) < len(snapshot.scenario.vms):
                    raise ValueError("Invalid VM assignment")
                engine.release_assigned(task, int(assignments[task.id]))
        end = max(t.deadline for t in batch)
        engine.advance(end)
        return engine.score(batch, snapshot.now, end)


class Predictor:
    """Fast queue-availability heuristic; the exact evaluator ranks full plans.

The heuristic reserves a candidate's entire route. Existing active/queued
services are included; their subsequent phases are left to exact replay.
"""
    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot
        self.available: dict[str, list[float]] = {}
        for key, state in snapshot.resources.items():
            availability = [end for _, end in state.active.values()] + [snapshot.now] * (state.capacity - len(state.active))
            for task_id in state.queue:
                rt = snapshot.tasks[task_id]
                phase = route(rt.task, rt.vm, snapshot.scenario)[rt.phase]
                index = int(np.argmin(availability))
                availability[index] += service_time(rt.task, rt.vm, phase, snapshot.scenario)
            self.available[key] = availability

    def predict(self, task: Task, vm: int, commit: bool = False) -> float:
        local = {}
        time = max(task.release, self.snapshot.now)
        for phase in route(task, vm, self.snapshot.scenario):
            key = resource_for(task, vm, phase)
            values = local.setdefault(key, list(self.available[key]))
            index = int(np.argmin(values))
            time = max(time, values[index]) + service_time(task, vm, phase, self.snapshot.scenario)
            values[index] = time
            time += propagation_time(task, vm, phase, self.snapshot.scenario)
        if commit:
            self.available.update(local)
        return max(1e-12, time-task.release)

    def greedy(self, batch: list[Task]) -> dict[int, int]:
        mapping = {}
        for task in batch:
            vm = min(range(len(self.snapshot.scenario.vms)), key=lambda v: (self.predict(task, v), v))
            mapping[task.id] = vm
            self.predict(task, vm, commit=True)
        return mapping

    def hints(self, batch: list[Task]) -> np.ndarray:
        return np.array([[self.predict(t, vm) for vm in range(len(self.snapshot.scenario.vms))] for t in batch])
