from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import os
import platform
import sys

import pandas as pd
import yaml

from . import __version__
from .algorithms import ALGORITHMS, Scheduler
from .analysis import analyze, plot_run
from .config import build_scenario, load_config, merge
from .simulation import Simulation
from .workload import generate, load_trace, save_trace


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_digest() -> str:
    root = Path(__file__).parent
    hasher = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        hasher.update(path.name.encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def dependencies() -> dict[str, str]:
    return {name: version(name) for name in ("simpy", "numpy", "pandas", "matplotlib", "PyYAML")}


def _json(path: Path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _csv(path: Path, rows, columns=None):
    temporary = path.with_suffix(".csv.tmp")
    pd.DataFrame(rows, columns=columns if not rows else None).to_csv(temporary, index=False)
    temporary.replace(path)


def run_one(config: dict, algorithm: str, seed: int, trace_path: str, directory: str,
            resume: bool = True, workers: int = 1) -> dict:
    directory, trace_path = Path(directory), Path(trace_path).resolve()
    scenario = build_scenario(config)
    # Names are identifiers, never filesystem expressions from configuration.
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in scenario.name):
        raise ValueError("Scenario name must contain only ASCII letters, digits, '_' and '-'")
    meta = dict(schema_version=1, package_version=__version__, config=scenario.config, algorithm=algorithm,
                seed=seed, trace_sha256=digest(trace_path), code_sha256=code_digest(), dependencies=dependencies(),
                python=platform.python_version(), workers=workers)
    signature = hashlib.sha256(json.dumps(meta, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    marker = directory/"_SUCCESS.json"
    if resume and marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("signature") == signature and all((directory/p).is_file() and digest(directory/p) == h for p, h in saved["files"].items()):
            return json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    result = Simulation().run(scenario, load_trace(trace_path), Scheduler(algorithm), seed)
    result.summary.update(trace_sha256=meta["trace_sha256"], code_sha256=meta["code_sha256"],
        config_sha256=hashlib.sha256(json.dumps(scenario.config, sort_keys=True).encode()).hexdigest())
    _csv(directory/"tasks.csv", result.tasks, ["task_id", "release", "finished", "status", "failure_reason", "latency_s", "measured", "tier"])
    _csv(directory/"decisions.csv", result.decisions, ["decision", "time", "batch_size", "evaluations", "search_s", "predicted_cost"])
    _csv(directory/"convergence.csv", result.convergence, ["decision", "time", "evaluation", "best_cost"])
    _csv(directory/"timeseries.csv", result.timeseries, ["time", "resource", "utilization", "queue_length", "active", "capacity"])
    _json(directory/"summary.json", result.summary)
    (directory/"config.yaml").write_text(yaml.safe_dump(scenario.config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    meta.update(trace_path=os.path.relpath(trace_path, directory), platform=platform.platform(), logical_cpus=os.cpu_count(), signature=signature)
    _json(directory/"metadata.json", meta)
    plot_run(directory)
    names = ["tasks.csv", "decisions.csv", "convergence.csv", "timeseries.csv", "summary.json", "config.yaml", "metadata.json"]
    names.extend(str(p.relative_to(directory)) for p in sorted((directory/"plots").glob("*")))
    _json(marker, dict(signature=signature, files={name: digest(directory/name) for name in names}))
    return result.summary


def prepare_trace(config: dict, seed: int, output: Path) -> Path:
    scenario = build_scenario(config)
    trace = generate(scenario, seed)
    trace_hash = hashlib.sha256(json.dumps([asdict(t) for t in trace], sort_keys=True).encode()).hexdigest()[:12]
    path = output/"traces"/scenario.name/f"seed_{seed}_{trace_hash}.jsonl"
    # Regenerate deterministically instead of trusting a truncated prior trace.
    save_trace(path, trace)
    return path.resolve()


def compare(configs: list[dict], output: Path, seeds=None, algorithms=None, workers=1, resume=True):
    if workers < 1:
        raise ValueError("workers must be positive")
    output = output.resolve()
    jobs = []
    identities = set()
    for config in configs:
        scenario = build_scenario(config)
        if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in scenario.name):
            raise ValueError("Invalid scenario name")
        chosen_seeds = seeds if seeds is not None else scenario.config["experiment"]["seeds"]
        chosen_algorithms = algorithms if algorithms is not None else scenario.config["experiment"]["algorithms"]
        for seed in chosen_seeds:
            if not isinstance(seed, int) or seed < 0:
                raise ValueError("Seeds must be nonnegative integers")
            trace = prepare_trace(scenario.config, seed, output)
            for algorithm in chosen_algorithms:
                if algorithm not in ALGORITHMS:
                    raise ValueError(f"Unknown algorithm {algorithm}")
                identity = scenario.name, seed, algorithm
                if identity in identities:
                    raise ValueError(f"Duplicate experiment {identity}")
                identities.add(identity)
                directory = output/"runs"/scenario.name/f"seed_{seed}"/algorithm
                jobs.append((scenario.config, algorithm, seed, str(trace), str(directory), resume, workers))
    summaries = []
    if workers == 1:
        for i, job in enumerate(jobs, 1):
            result = run_one(*job)
            summaries.append(result)
            print(f"[{i}/{len(jobs)}] {result['scenario']} seed={result['seed']} {result['algorithm']}: cost={result['total']:.6f}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_one, *job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                summaries.append(result)
                print(f"[{i}/{len(jobs)}] {result['scenario']} seed={result['seed']} {result['algorithm']}: cost={result['total']:.6f}", flush=True)
    if not summaries:
        raise ValueError("No experiments selected")
    frame = pd.DataFrame(summaries)
    repetitions = int(build_scenario(configs[0]).config["experiment"]["bootstrap_samples"])
    analyze(frame, output/"analysis", repetitions)
    from .analysis import plot_convergence_comparisons
    plot_convergence_comparisons(output)
    return frame


def suite_configs(manifest: str | Path) -> list[dict]:
    manifest = Path(manifest).resolve()
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    configs = []
    for entry in raw["scenarios"]:
        config = load_config(manifest.parent/entry["config"])
        if "scan" not in entry:
            configs.append(config)
            continue
        # One dotted parameter per scan; no accidental factorial combinations.
        key = entry["scan"]["parameter"]
        for value in entry["scan"]["values"]:
            change = value
            for part in reversed(key.split(".")):
                change = {part: change}
            variant = merge(config, change)
            if "names" in entry["scan"]:
                index = entry["scan"]["values"].index(value)
                name = entry["scan"]["names"][index]
            else:
                label = str(value).replace(".", "p")
                name = f"{config['name']}_{key.split('.')[-1]}_{label}"
            variant["name"] = name
            variant["_scan"] = dict(group=f"{config['name']}_{key.replace('.', '_')}", parameter=key, value=value)
            configs.append(build_scenario(variant).config)
    return configs


def reanalyze(output: Path, repetitions=10000):
    rows = []
    for marker in sorted((output/"runs").rglob("_SUCCESS.json")):
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if not all((marker.parent/p).exists() and digest(marker.parent/p) == h for p, h in saved["files"].items()):
            raise ValueError(f"Incomplete or modified run: {marker.parent}")
        rows.append(json.loads((marker.parent/"summary.json").read_text(encoding="utf-8")))
    if not rows:
        raise ValueError("No completed runs found")
    result = analyze(pd.DataFrame(rows), output/"analysis", repetitions)
    from .analysis import plot_convergence_comparisons
    plot_convergence_comparisons(output)
    return result
