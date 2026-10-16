from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import json

import numpy as np

from .models import Scenario, Task, task_dict


def rng_for(seed: int, stream: str) -> np.random.Generator:
    # Python's randomized hash() must never determine experiment seeds.
    digest = hashlib.sha256(stream.encode()).digest()
    words = [int.from_bytes(digest[i:i+4], "little") for i in range(0, 16, 4)]
    return np.random.default_rng(np.random.SeedSequence([seed, *words]))


def _task(index: int, release: float, ap: int, device: int, app: dict) -> Task:
    return Task(index, round(release, 12), ap, app["name"], float(app["mi"]),
                int(round(float(app["input_kb"]) * 1000)), int(round(float(app["output_kb"]) * 1000)),
                float(app["max_delay"]), device)


def generate(scenario: Scenario, seed: int) -> list[Task]:
    cfg = scenario.config["workload"]
    apps = cfg["apps"]
    probabilities = [a["share"] for a in apps]
    rng = rng_for(seed, "workload")
    tasks = []
    if scenario.mode == "static":
        for i in range(cfg["task_count"]):
            app = apps[int(rng.choice(len(apps), p=probabilities))]
            tasks.append(_task(i, rng.uniform(0, cfg["release_window"]), i % scenario.aps, i, app))
    else:
        # Integrate the piecewise intensity exactly; don't stretch interarrival
        # intervals across burst boundaries with an outdated rate.
        boundaries = {0., scenario.duration}
        for b in cfg["bursts"]:
            boundaries.update((float(b["start"]), float(b["end"])))
        boundaries = sorted(boundaries)
        segments = []
        for start, end in zip(boundaries, boundaries[1:]):
            multiplier = next((b["multiplier"] for b in cfg["bursts"] if b["start"] <= start < b["end"]), 1.)
            segments.append((start, end, float(multiplier)))
        for device in range(cfg["devices"]):
            app = apps[int(rng.choice(len(apps), p=probabilities))]
            for start, end, multiplier in segments:
                t = start + float(rng.exponential(app["interval"] / multiplier))
                while t < end:
                    tasks.append(_task(len(tasks), t, device % scenario.aps, device, app))
                    t += float(rng.exponential(app["interval"] / multiplier))
    tasks.sort(key=lambda t: (t.release, t.id))
    return [replace(task, id=i) for i, task in enumerate(tasks)]


def validate_trace(trace: list[Task], scenario: Scenario) -> None:
    if len({t.id for t in trace}) != len(trace):
        raise ValueError("Trace task IDs must be unique")
    for t in trace:
        numbers = (t.release, t.mi, t.max_delay, t.input_bytes, t.output_bytes)
        if any(not np.isfinite(x) for x in numbers) or t.release < 0 or t.mi <= 0 or t.max_delay <= 0 or min(t.input_bytes, t.output_bytes) < 0:
            raise ValueError(f"Invalid task {t.id}")
        if not isinstance(t.id, int) or isinstance(t.id, bool) or t.id < 0 or not isinstance(t.ap, int) or isinstance(t.ap, bool) or not 0 <= t.ap < scenario.aps:
            raise ValueError(f"Invalid ID/AP for task {t.id}")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (t.input_bytes, t.output_bytes)):
            raise ValueError("Trace sizes must be integer byte counts")
        if scenario.mode == "dynamic" and t.release >= scenario.duration:
            raise ValueError("Dynamic trace arrivals must precede generation stop")


def save_trace(path: Path, trace: list[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for task in trace:
            stream.write(json.dumps(task_dict(task), ensure_ascii=False) + "\n")


def load_trace(path: str | Path) -> list[Task]:
    with Path(path).open(encoding="utf-8") as stream:
        return sorted([Task(**json.loads(line)) for line in stream if line.strip()], key=lambda t: (t.release, t.id))
