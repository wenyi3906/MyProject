from dataclasses import replace

import numpy as np
import pytest

from cloudedge.algorithms import Scheduler
from cloudedge.config import load_config, build_scenario, merge
from cloudedge.engine import Engine, Evaluator
from cloudedge.simulation import Simulation
from cloudedge.workload import generate, rng_for, save_trace, load_trace
from conftest import scenario, task


def test_report_units_and_deterministic_trace_roundtrip(tmp_path):
    s = build_scenario(load_config("configs/smoke_static.yaml"))
    trace = generate(s, 0)
    assert trace == generate(s, 0)
    assert trace != generate(s, 1)
    ar = next(t for t in trace if t.app == "ar")
    assert ar.input_bytes == 1500000 and ar.output_bytes == 25000
    save_trace(tmp_path/"trace.jsonl", trace)
    assert load_trace(tmp_path/"trace.jsonl") == trace


def test_piecewise_poisson_burst_rate_and_bounds():
    apps = [dict(name="test", share=1, interval=1, max_delay=1, input_kb=0, output_kb=0, mi=1)]
    s = scenario(mode="dynamic", simulation={"duration": 10}, workload={"devices": 1000, "apps": apps, "bursts": [{"start": 3, "end": 6, "multiplier": 3}]})
    trace = generate(s, 3)
    times = np.array([t.release for t in trace])
    baseline = ((times < 3) | (times >= 6)).sum()/7
    burst = ((times >= 3) & (times < 6)).sum()/3
    assert 2.8 < burst/baseline < 3.2
    assert times.min() >= 0 and times.max() < 10
    assert len({t.id for t in trace}) == len(trace)


def test_static_actual_score_equals_full_trace_evaluation():
    s = scenario()
    trace = [task(0, release=.1, up=10000, delay=2), task(1, release=.2, delay=.5), task(2, release=.2, mi=5)]
    class Fixed(Scheduler):
        def solve(self, batch, snapshot, evaluator, budget, rng):
            result = super().solve(batch, snapshot, evaluator, budget, rng)
            return result
    scheduler = Fixed("ga")
    snapshot = Engine(s).snapshot()
    expected = Scheduler("ga").solve(trace, snapshot, Evaluator(), s.budget, rng_for(3, "algorithm:ga"))
    result = Simulation().run(s, trace, scheduler, 3)
    assert result.summary["total"] == pytest.approx(expected.score.total)


def test_dynamic_window_edf_limit_expiration_warmup_and_drain():
    s = scenario(mode="dynamic", simulation={"duration": .5, "warmup": .1, "window": .1, "max_batch": 1})
    trace = [task(0, release=.01, mi=1, delay=.05), task(1, release=.11, mi=1, delay=.5),
             task(2, release=.11, mi=1, delay=.3), task(3, release=.49, mi=10, delay=.5)]
    result = Simulation().run(s, trace, Scheduler("round_robin"), 0)
    rows = result.tasks
    assert rows[0]["failure_reason"] == "scheduler_wait"
    assert not rows[0]["measured"]
    assert rows[2]["dispatched"] == .2
    assert rows[1]["dispatched"] == .3
    assert rows[3]["finished"] > .5
    assert result.summary["succeeded"]+result.summary["failed"] == result.summary["measured"] == 3
    assert all(row["batch_size"] == 1 for row in result.decisions)
    for row in rows:
        duration = row["scheduler_wait_s"] + sum(v for k, v in row.items() if k.endswith(("_queue_s", "_service_s", "_propagation_s")))
        assert duration == pytest.approx(row["latency_s"])


def test_future_trace_changes_do_not_change_current_decision():
    s = scenario(mode="dynamic", simulation={"duration": 3})
    first = task(0, release=.01, mi=100, delay=1)
    early = Simulation().run(s, [first], Scheduler("ga"), 5)
    later = Simulation().run(s, [first, task(1, release=2, mi=10000)], Scheduler("ga"), 5)
    assert early.tasks[0] == later.tasks[0]
    assert early.convergence[:10] == later.convergence[:10]


def test_empty_trace_is_well_defined():
    result = Simulation().run(scenario(mode="dynamic"), [], Scheduler("ga"))
    assert result.summary["measured"] == 0
    assert result.summary["mean_latency_s"] is None
    assert result.summary["total"] == 0


@pytest.mark.parametrize("changes", [
    {"optimization": {"weights": [.5, .5, .5]}},
    {"simulation": {"window": 0}},
    {"network": {"wlan_mbps": 0}},
    {"topology": {"edge_mips": [100, 200]}},
    {"workload": {"bursts": [{"start": 1, "end": 4, "multiplier": 0}]}},
])
def test_invalid_configs_fail_early(changes):
    with pytest.raises(ValueError):
        scenario(**changes)


def test_benchmark_suites_and_capacity_preservation():
    from cloudedge.experiments import suite_configs
    assert len(suite_configs("configs/benchmark_suite.yaml")) == 12
    assert len(suite_configs("configs/sensitivity_suite.yaml")) == 9
    original = build_scenario(load_config("configs/base.yaml"))
    hetero = build_scenario(load_config("configs/heterogeneous.yaml"))
    assert sum(v.mips for v in original.vms if v.tier == "edge") == sum(v.mips for v in hetero.vms if v.tier == "edge")


def test_configuration_typos_and_inheritance_cycles_fail(tmp_path):
    with pytest.raises(ValueError, match="Unknown configuration key"):
        scenario(workload={"devcies": 100})
    (tmp_path/"a.yaml").write_text("extends: b.yaml", encoding="utf-8")
    (tmp_path/"b.yaml").write_text("extends: a.yaml", encoding="utf-8")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path/"a.yaml")
