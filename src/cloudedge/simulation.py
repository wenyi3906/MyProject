from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np

from .algorithms import Scheduler
from .engine import Engine, Evaluator
from .models import RunResult, Scenario, Task
from .workload import rng_for, validate_trace


class Simulation:
    def run(self, scenario: Scenario, trace: list[Task], scheduler: Scheduler, seed: int = 0) -> RunResult:
        validate_trace(trace, scenario)
        trace = sorted(trace, key=lambda t: (t.release, t.id))
        engine, evaluator = Engine(scenario), Evaluator()
        rng = rng_for(seed, f"algorithm:{scheduler.name}")
        decisions, convergence, samples = [], [], []
        static = scenario.mode == "static"
        measure_start = 0. if static else scenario.warmup
        measure_end = max((t.deadline for t in trace), default=scenario.duration) if static else scenario.duration
        cohort = trace if static else [t for t in trace if measure_start <= t.release < measure_end]

        def solve(batch):
            index = len(decisions)
            result = scheduler.solve(batch, engine.snapshot(), evaluator, scenario.budget, rng)
            decisions.append(dict(decision=index, time=engine.now, batch_size=len(batch), evaluations=result.evaluations,
                search_s=result.elapsed_s, predicted_cost=result.score.total,
                predicted_failure_rate=result.score.failure_rate, predicted_delay=result.score.normalized_delay,
                predicted_imbalance=result.score.imbalance))
            convergence.extend(dict(decision=index, time=engine.now, **row) for row in result.convergence)
            return result.assignments

        if static and trace:
            assignment = solve(trace)
            for task in trace:
                engine.release_assigned(task, assignment[task.id])
        arrival_index = len(trace) if static else 0
        sample_index = 0
        next_tick = math.inf

        while True:
            arrival = trace[arrival_index].release if arrival_index < len(trace) else math.inf
            sample = round(sample_index*scenario.sample_interval, 12)
            if sample > measure_end:
                sample = math.inf
            internal = engine.env.peek() if engine.events else math.inf
            now = min(arrival, sample, next_tick, internal)
            if math.isinf(now):
                break
            engine.advance(now)
            if not static:
                while arrival_index < len(trace) and trace[arrival_index].release <= now:
                    engine.add_task(trace[arrival_index])
                    arrival_index += 1
                pending = [engine.tasks[key].task for key in engine.pending]
                if pending and math.isinf(next_tick):
                    next_tick = round(math.ceil((now-1e-10)/scenario.window)*scenario.window, 12)
                    next_tick = max(now, next_tick)
                if next_tick <= now:
                    pending.sort(key=lambda t: (t.deadline, t.id))
                    batch = pending[:scenario.max_batch]
                    if batch:
                        assignment = solve(batch)
                        for task in batch:
                            engine.dispatch(task.id, assignment[task.id])
                    remaining = bool(engine.pending)
                    next_tick = round(now+scenario.window, 12) if remaining else math.inf
            if sample <= now:
                for key, resource in engine.resources.items():
                    samples.append(dict(time=now, resource=key, utilization=len(resource.active)/resource.capacity,
                                        queue_length=len(resource.queue), active=len(resource.active), capacity=resource.capacity))
                samples.append(dict(time=now, resource="scheduler", utilization=0., active=0, capacity=1,
                                    queue_length=len(engine.pending)))
                sample_index += 1

        rows = engine.task_rows()
        measured_ids = {t.id for t in cohort}
        for row in rows:
            row["measured"] = row["task_id"] in measured_ids
        measured = [row for row in rows if row["measured"]]
        success = [row for row in measured if row["status"] == "success"]
        latencies = [row["latency_s"] for row in success]
        score = engine.score(cohort, measure_start, measure_end)
        edge = [engine.utilization(f"vm_{v.id}", measure_start, measure_end) for v in scenario.vms if v.tier == "edge"]
        cloud = [engine.utilization(f"vm_{v.id}", measure_start, measure_end) for v in scenario.vms if v.tier == "cloud"]
        searches = [d["search_s"] for d in decisions]
        summary = dict(scenario=scenario.name, mode=scenario.mode, algorithm=scheduler.name, seed=seed,
            group=scenario.config.get("_scan", {}).get("group", scenario.name),
            scan_parameter=scenario.config.get("_scan", {}).get("parameter"),
            scan_value=scenario.config.get("_scan", {}).get("value"),
            devices=scenario.config["workload"]["devices"], task_count=scenario.config["workload"]["task_count"],
            window_s=scenario.window, evaluation_budget=scenario.budget,
            measurement_start=measure_start, measurement_end=measure_end, finished_at=engine.now,
            generated=len(trace), measured=len(cohort), succeeded=len(success), failed=len(cohort)-len(success),
            **asdict(score), mean_latency_s=float(np.mean(latencies)) if latencies else None,
            p95_latency_s=float(np.quantile(latencies, .95)) if latencies else None,
            p99_latency_s=float(np.quantile(latencies, .99)) if latencies else None,
            # Throughput counts returns within the fixed measurement interval;
            # cohort success includes drain completions, so both are exported.
            throughput_per_s=sum(measure_start <= rt.finished <= measure_end and rt.status == "success" for rt in engine.tasks.values())/(measure_end-measure_start),
            cohort_success_per_s=len(success)/(measure_end-measure_start),
            edge_utilization=float(np.mean(edge)), cloud_utilization=float(np.mean(cloud)),
            edge_load_std=float(np.std(edge)), cloud_load_std=float(np.std(cloud)),
            cloud_fraction=sum(r["tier"] == "cloud" for r in measured)/len(measured) if measured else 0.,
            decisions=len(decisions), evaluations=sum(d["evaluations"] for d in decisions),
            search_total_s=sum(searches), search_mean_s=float(np.mean(searches)) if searches else 0.,
            search_p95_s=float(np.quantile(searches, .95)) if searches else 0.)
        return RunResult(summary, rows, decisions, convergence, samples)
