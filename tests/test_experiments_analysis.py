from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from cloudedge.analysis import METRICS, statistics
from cloudedge.cli import main
from cloudedge.experiments import compare, prepare_trace, reanalyze, run_one
from cloudedge.workload import save_trace
from conftest import scenario, task


def test_bootstrap_paired_differences_match_seed_not_row_order():
    rows = []
    for name, offset in (("ga", 0.), ("sa", .1)):
        for seed in (0, 1, 2):
            row = {metric: .2+seed*.01+offset for metric in METRICS}
            row.update(scenario="test", algorithm=name, seed=seed)
            rows.append(row)
    rows[0]["mean_latency_s"] = None
    frame = pd.DataFrame(rows).sample(frac=1, random_state=3)
    stats, paired = statistics(frame, 300)
    cost = paired[paired.metric == "total"].iloc[0]
    assert cost.mean_difference == pytest.approx(-.1)
    assert cost.ci_low == pytest.approx(-.1) and cost.ci_high == pytest.approx(-.1)
    assert stats[(stats.algorithm == "ga") & (stats.metric == "mean_latency_s")].iloc[0].valid_runs == 2
    assert statistics(frame, 300)[0].equals(stats)


def test_run_artifacts_resume_checksum_and_replay(tmp_path):
    s = scenario(optimization={"static_budget": 5})
    trace = tmp_path/"trace.jsonl"
    save_trace(trace, [task(0, mi=10), task(1, mi=20)])
    directory = tmp_path/"original"
    first = run_one(s.config, "ga", 0, str(trace), str(directory))
    marker = directory/"_SUCCESS.json"
    initial_time = marker.stat().st_mtime_ns
    assert run_one(s.config, "ga", 0, str(trace), str(directory)) == first
    assert marker.stat().st_mtime_ns == initial_time
    rows = pd.read_csv(directory/"tasks.csv")
    assert len(rows) == first["generated"]
    (directory/"tasks.csv").write_text("corrupt", encoding="utf-8")
    repaired = run_one(s.config, "ga", 0, str(trace), str(directory))
    assert len(pd.read_csv(directory/"tasks.csv")) == 2
    assert repaired["total"] == first["total"]
    main(["replay", "--run", str(directory), "--out", str(tmp_path/"replay")])
    replay = json.loads((tmp_path/"replay"/"run"/"summary.json").read_text())
    for key, value in repaired.items():
        if not key.startswith("search_"):
            assert replay[key] == value
    pd.testing.assert_frame_equal(pd.read_csv(directory/"tasks.csv"), pd.read_csv(tmp_path/"replay"/"run"/"tasks.csv"))


def test_batch_outputs_and_analysis_reconstruction(tmp_path):
    s = scenario(workload={"task_count": 3, "release_window": .1}, optimization={"static_budget": 4})
    frame = compare([s.config], tmp_path, seeds=[0, 1], algorithms=["ga", "sa"])
    assert len(frame) == 4
    assert (tmp_path/"analysis"/"report.md").is_file()
    assert (tmp_path/"analysis"/"plots"/"test_comparison.svg").is_file()
    for seed in (0, 1):
        first = json.loads((tmp_path/f"runs/test/seed_{seed}/ga/metadata.json").read_text())
        second = json.loads((tmp_path/f"runs/test/seed_{seed}/sa/metadata.json").read_text())
        assert first["trace_sha256"] == second["trace_sha256"]
    before = pd.read_csv(tmp_path/"analysis"/"statistics.csv")
    reanalyze(tmp_path, repetitions=100)
    pd.testing.assert_frame_equal(before, pd.read_csv(tmp_path/"analysis"/"statistics.csv"))


def test_all_failed_statistics_and_plot_are_valid(tmp_path):
    from cloudedge.analysis import analyze
    frame = pd.DataFrame([{**{metric: 0. for metric in METRICS}, "scenario": "failed", "algorithm": name,
                          "seed": seed, "failure_rate": 1., "mean_latency_s": None, "p95_latency_s": None, "p99_latency_s": None}
                         for name in ("ga", "sa") for seed in (0, 1)])
    stats, paired = analyze(frame, tmp_path, 100)
    assert stats[stats.metric == "mean_latency_s"].valid_runs.eq(0).all()
    assert paired[paired.metric == "mean_latency_s"].paired_runs.eq(0).all()
    assert (tmp_path/"plots"/"failed_comparison.png").is_file()


def test_enumeration_cli_reports_gaps(tmp_path):
    main(["exhaustive", "--config", "configs/tiny.yaml", "--seed", "0", "--out", str(tmp_path)])
    gaps = pd.read_csv(tmp_path/"optimality_gaps.csv")
    assert len(gaps) == 7
    assert gaps.absolute_gap.ge(-1e-12).all()
    assert json.loads((tmp_path/"optimum.json").read_text())["combinations"] == 64


def test_statistics_rejects_mismatched_traces():
    rows = [{**{metric: .2 for metric in METRICS}, "scenario": "test", "algorithm": name, "seed": 0, "trace_sha256": name}
            for name in ("ga", "sa")]
    with pytest.raises(ValueError, match="trace_sha256"):
        statistics(pd.DataFrame(rows))


def test_parallel_batch_and_scan_plot(tmp_path):
    from cloudedge.config import merge
    configurations = []
    for count in (2, 3):
        s = scenario(name=f"scan_{count}", workload={"task_count": count, "release_window": .1}, optimization={"static_budget": 2})
        configurations.append(merge(s.config, {"_scan": {"group": "task_count", "parameter": "workload.task_count", "value": count}}))
    frame = compare(configurations, tmp_path, seeds=[0], algorithms=["ga"], workers=2)
    assert len(frame) == 2
    assert (tmp_path/"analysis"/"plots"/"task_count_trend.png").is_file()
