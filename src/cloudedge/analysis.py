from __future__ import annotations

from itertools import combinations
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd()/".cache"/"matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .workload import rng_for


METRICS = ("total", "failure_rate", "mean_latency_s", "p95_latency_s", "p99_latency_s",
           "normalized_delay", "imbalance", "throughput_per_s", "edge_utilization", "cloud_utilization",
           "edge_load_std", "cloud_fraction", "search_mean_s", "search_p95_s")


def bootstrap(values, rng, repetitions=10000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean
    # Limit peak allocation when users increase repetition/seed counts.
    means = []
    for start in range(0, repetitions, 1000):
        indices = rng.integers(len(values), size=(min(1000, repetitions-start), len(values)))
        means.extend(values[indices].mean(axis=1))
    lower, upper = np.quantile(means, [.025, .975])
    return mean, float(lower), float(upper)


def statistics(frame: pd.DataFrame, repetitions: int = 10000) -> tuple[pd.DataFrame, pd.DataFrame]:
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    if frame.duplicated(["scenario", "algorithm", "seed"]).any():
        raise ValueError("Duplicate scenario/algorithm/seed records cannot be pooled")
    for fingerprint in ("trace_sha256", "config_sha256", "code_sha256"):
        if fingerprint in frame and frame.groupby(["scenario", "seed"])[fingerprint].nunique().gt(1).any():
            raise ValueError(f"Paired comparison requires matching {fingerprint} within each scenario/seed")
    stats, paired = [], []
    for (scenario, algorithm), group in frame.groupby(["scenario", "algorithm"], sort=True):
        for metric in METRICS:
            mean, lower, upper = bootstrap(group[metric], rng_for(0, f"bootstrap:{scenario}:{algorithm}:{metric}"), repetitions)
            stats.append(dict(scenario=scenario, algorithm=algorithm, metric=metric,
                runs=len(group), valid_runs=int(group[metric].notna().sum()), mean=mean, ci_low=lower, ci_high=upper))
    for scenario, group in frame.groupby("scenario", sort=True):
        for first, second in combinations(sorted(group.algorithm.unique()), 2):
            joined = group[group.algorithm == first].merge(group[group.algorithm == second], on="seed", suffixes=("_a", "_b"))
            for metric in METRICS:
                values = pd.to_numeric(joined[f"{metric}_a"], errors="coerce") - pd.to_numeric(joined[f"{metric}_b"], errors="coerce")
                mean, lower, upper = bootstrap(values, rng_for(0, f"paired:{scenario}:{first}:{second}:{metric}"), repetitions)
                paired.append(dict(scenario=scenario, algorithm_a=first, algorithm_b=second, metric=metric,
                    paired_runs=int(values.notna().sum()), mean_difference=mean, ci_low=lower, ci_high=upper))
    return pd.DataFrame(stats), pd.DataFrame(paired, columns=["scenario", "algorithm_a", "algorithm_b", "metric", "paired_runs", "mean_difference", "ci_low", "ci_high"])


def _save(fig, directory: Path, name: str):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(directory/f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_statistics(stats: pd.DataFrame, output: Path):
    for scenario, group in stats.groupby("scenario", sort=True):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        panels = [("total", "Weighted cost (lower is better)"), ("failure_rate", "Failure rate"),
                  ("mean_latency_s", "Successful-task mean latency (s)"), ("imbalance", "Edge imbalance"),
                  ("edge_utilization", "Edge VM utilization"), ("search_mean_s", "Mean search time (s)")]
        for ax, (metric, title) in zip(axes.flat, panels):
            data = group[group.metric == metric].sort_values("algorithm")
            values = data["mean"].to_numpy(float)
            lower = values-data.ci_low.to_numpy(float)
            upper = data.ci_high.to_numpy(float)-values
            ax.bar(data.algorithm, values, yerr=np.maximum(0, np.vstack([lower, upper])), capsize=3, color="#3f83b5")
            ax.set_title(title, fontsize=11)
            ax.tick_params(axis="x", rotation=30)
            ax.grid(axis="y", alpha=.2)
            if metric in ("failure_rate", "imbalance", "edge_utilization"):
                ax.set_ylim(0, 1)
        fig.suptitle(f"{scenario}: run means and 95% bootstrap intervals", fontsize=14)
        _save(fig, output, f"{scenario}_comparison")


def plot_run(run_directory: Path):
    # Remove only the three known generated figures, so rerunning a now-empty
    # or all-successful trace cannot leave a stale failure/convergence chart.
    for name in ("convergence", "resources", "failure_phases"):
        for extension in ("png", "svg"):
            path = run_directory/"plots"/f"{name}.{extension}"
            if path.is_file():
                path.unlink()
    convergence = pd.read_csv(run_directory/"convergence.csv")
    output = run_directory/"plots"
    if len(convergence):
        first = convergence[convergence.decision == convergence.decision.min()]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(first.evaluation, first.best_cost, color="#3f83b5")
        ax.set(xlabel="Complete candidate evaluations", ylabel="Best weighted cost", title="First decision: convergence")
        ax.grid(alpha=.25)
        _save(fig, output, "convergence")
    series = pd.read_csv(run_directory/"timeseries.csv")
    if len(series):
        vm = series[series.resource.str.startswith("vm_")]
        grouped = vm.groupby("time")[["utilization", "queue_length"]].mean()
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
        axes[0].plot(grouped.index, grouped.utilization, color="#3f83b5")
        axes[0].set(ylabel="Mean VM utilization", ylim=(0, 1))
        axes[1].plot(grouped.index, grouped.queue_length, color="#d07c3f")
        axes[1].set(xlabel="Simulation time (s)", ylabel="Mean VM queue length")
        for ax in axes:
            ax.grid(alpha=.25)
        _save(fig, output, "resources")
    tasks = pd.read_csv(run_directory/"tasks.csv")
    failed = tasks[(tasks.measured == True) & (tasks.status == "failed")]
    if len(failed):
        reasons = failed.failure_reason.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(reasons.index, reasons.values, color="#d07c3f")
        ax.set(title="Measured task failures by phase", ylabel="Tasks")
        ax.tick_params(axis="x", rotation=30)
        _save(fig, output, "failure_phases")


def analyze(frame: pd.DataFrame, output: Path, repetitions=10000):
    output.mkdir(parents=True, exist_ok=True)
    stats, paired = statistics(frame, repetitions)
    frame.sort_values(["scenario", "seed", "algorithm"]).to_csv(output/"runs.csv", index=False)
    stats.to_csv(output/"statistics.csv", index=False)
    paired.to_csv(output/"paired_differences.csv", index=False)
    plot_statistics(stats, output/"plots")
    plot_scan_trends(frame, stats, output/"plots")
    report = ["# 实验报告", "", f"完成 {len(frame)} 次运行。区间按运行种子进行 bootstrap，不能据少量种子作稳定的算法排名。", "",
              "成功任务平均延迟应与失败率一起阅读；搜索耗时是实际计算时间，未加入仿真时钟。", ""]
    for scenario, group in stats.groupby("scenario", sort=True):
        report.extend([f"## {scenario}", "", "| 算法 | 代价均值 [95% CI] | 失败率 | 成功任务平均延迟/s |", "|---|---|---|---|"])
        for algorithm in sorted(group.algorithm.unique()):
            rows = group[group.algorithm == algorithm].set_index("metric")
            cost, fail, delay = (rows.loc[key] for key in ("total", "failure_rate", "mean_latency_s"))
            latency = "无成功任务" if not np.isfinite(delay["mean"]) else f"{delay['mean']:.6f}"
            report.append(f"| {algorithm} | {cost['mean']:.6f} [{cost.ci_low:.6f}, {cost.ci_high:.6f}] | {fail['mean']:.2%} | {latency} |")
        report.append("")
    report.extend(["配对比较见 `paired_differences.csv`：差值为算法 A 减算法 B，仅匹配同一场景、同一任务轨迹种子。", ""])
    (output/"report.md").write_text("\n".join(report), encoding="utf-8")
    return stats, paired


def plot_scan_trends(frame: pd.DataFrame, stats: pd.DataFrame, output: Path):
    if "scan_parameter" not in frame:
        return
    scanned = frame[frame.scan_parameter.notna()]
    for group_name, group in scanned.groupby("group", sort=True):
        cases = group[["scenario", "scan_value"]].drop_duplicates("scenario")
        scalar = all(isinstance(v, (int, float)) for v in cases.scan_value)
        if scalar:
            cases = cases.sort_values("scan_value")
        else:
            cases = cases.sort_values("scenario")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for ax, metric, title in zip(axes, ("failure_rate", "mean_latency_s", "total"),
                                      ("Failure rate", "Successful-task latency (s)", "Weighted cost")):
            for algorithm in sorted(group.algorithm.unique()):
                data = stats[(stats.algorithm == algorithm) & (stats.metric == metric)].set_index("scenario")
                data = data.loc[cases.scenario]
                x = cases.scan_value.to_numpy(float) if scalar else np.arange(len(cases))
                values = data["mean"].to_numpy(float)
                ax.errorbar(x, values, yerr=np.maximum(0, np.vstack([values-data.ci_low, data.ci_high-values])),
                            marker="o", markersize=4, capsize=2, label=algorithm)
            ax.set_title(title)
            ax.set_xlabel(group.scan_parameter.iloc[0])
            ax.grid(alpha=.25)
            if not scalar:
                ax.set_xticks(np.arange(len(cases)), cases.scenario, rotation=20)
            if metric == "failure_rate":
                ax.set_ylim(0, 1)
        axes[-1].legend(fontsize=8)
        _save(fig, output, f"{group_name}_trend")


def plot_convergence_comparisons(root: Path):
    runs = root/"runs"
    if not runs.exists():
        return
    for scenario in sorted(p for p in runs.iterdir() if p.is_dir()):
        seeds = sorted((p for p in scenario.iterdir() if p.is_dir() and p.name.startswith("seed_")), key=lambda p: int(p.name[5:]))
        if not seeds:
            continue
        series = []
        for path in sorted(seeds[0].glob("*/convergence.csv")):
            if not (path.parent/"_SUCCESS.json").exists():
                continue
            frame = pd.read_csv(path)
            if len(frame):
                series.append((path.parent.name, frame[frame.decision == frame.decision.min()]))
        if not series:
            continue
        last = max(int(frame.evaluation.max()) for _, frame in series)
        fig, ax = plt.subplots(figsize=(9, 5))
        for algorithm, frame in series:
            if len(frame) == 1:
                ax.plot([1, max(2, last)], [frame.best_cost.iloc[0]]*2, "--", label=algorithm)
            else:
                ax.plot(frame.evaluation, frame.best_cost, label=algorithm)
        ax.set(title=f"{scenario.name}: first decision, {seeds[0].name}",
               xlabel="Complete candidate evaluations", ylabel="Best weighted cost")
        ax.legend()
        ax.grid(alpha=.25)
        _save(fig, root/"analysis"/"plots", f"{scenario.name}_convergence")
