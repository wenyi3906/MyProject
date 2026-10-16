from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    id: int
    release: float
    ap: int
    app: str
    mi: float
    input_bytes: int
    output_bytes: int
    max_delay: float
    device: int = -1

    @property
    def deadline(self) -> float:
        return round(self.release + self.max_delay, 12)


@dataclass(frozen=True)
class VM:
    id: int
    host: str
    tier: str
    mips: float
    slots: int = 1


@dataclass(frozen=True)
class Link:
    id: str
    mbps: float
    propagation: float


@dataclass(frozen=True)
class Scenario:
    name: str
    mode: str
    duration: float
    warmup: float
    window: float
    max_batch: int
    sample_interval: float
    budget: int
    weights: tuple[float, float, float]
    vms: tuple[VM, ...]
    links: tuple[Link, ...]
    aps: int
    config: dict[str, Any] = field(compare=False, repr=False)


@dataclass
class StageTiming:
    requested: float
    started: float | None = None
    service_end: float | None = None
    finished: float | None = None

    def durations(self, terminal: float) -> tuple[float, float, float]:
        end = self.finished if self.finished is not None else terminal
        waiting = max(0.0, (self.started if self.started is not None else end) - self.requested)
        service = 0.0 if self.started is None else max(0.0, (self.service_end if self.service_end is not None else end) - self.started)
        propagation = 0.0 if self.service_end is None else max(0.0, end - self.service_end)
        return waiting, service, propagation


@dataclass
class RuntimeTask:
    task: Task
    vm: int | None = None
    dispatched: float | None = None
    status: str = "pending"
    stage: str = "scheduler_wait"
    phase: int = 0
    finished: float | None = None
    failure_reason: str | None = None
    timings: dict[str, StageTiming] = field(default_factory=dict)
    events: set[int] = field(default_factory=set)


@dataclass
class ResourceState:
    id: str
    capacity: int = 1
    queue: list[int] = field(default_factory=list)
    # task -> (service start, scheduled completion)
    active: dict[int, tuple[float, float]] = field(default_factory=dict)
    intervals: list[tuple[float, float]] = field(default_factory=list)


@dataclass(frozen=True)
class EventRecord:
    id: int
    time: float
    priority: int
    kind: str
    task_id: int
    data: Any = None


@dataclass(frozen=True)
class Snapshot:
    now: float
    scenario: Scenario
    tasks: dict[int, RuntimeTask]
    resources: dict[str, ResourceState]
    events: tuple[EventRecord, ...]
    next_id: int


@dataclass(frozen=True)
class Score:
    failure_rate: float
    normalized_delay: float
    imbalance: float
    total: float


@dataclass
class ScheduleResult:
    assignments: dict[int, int]
    score: Score
    evaluations: int
    elapsed_s: float
    convergence: list[dict[str, float | int]]


@dataclass
class RunResult:
    summary: dict[str, Any]
    tasks: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    convergence: list[dict[str, Any]]
    timeseries: list[dict[str, Any]]


def task_dict(task: Task) -> dict[str, Any]:
    return asdict(task)

