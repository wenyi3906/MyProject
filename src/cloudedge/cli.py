from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import json
import time

import pandas as pd

from .algorithms import ALGORITHMS, Scheduler
from .config import build_scenario, load_config
from .engine import Engine, Evaluator
from .experiments import compare, prepare_trace, reanalyze, run_one, suite_configs
from .workload import generate, load_trace, rng_for, save_trace, validate_trace


def parser():
    root = argparse.ArgumentParser(prog="cloudedge", description="Reproducible cloud-edge scheduling experiments")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("generate", "run", "compare", "exhaustive"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name != "compare":
            command.add_argument("--seed", type=int, default=100)
        command.add_argument("--out", type=Path, required=True)
        if name in ("run", "exhaustive"):
            command.add_argument("--trace", type=Path)
        if name == "run":
            command.add_argument("--algorithm", choices=ALGORITHMS, default="ga")
            command.add_argument("--no-resume", action="store_true")
        if name == "exhaustive":
            command.add_argument("--max-combinations", type=int, default=1000000)
    suite = sub.add_parser("suite")
    suite.add_argument("--manifest", type=Path, required=True)
    suite.add_argument("--out", type=Path, required=True)
    for command in (sub.choices["compare"], suite):
        command.add_argument("--seeds", type=int, nargs="+")
        command.add_argument("--algorithms", choices=ALGORITHMS, nargs="+")
        command.add_argument("--workers", type=int, default=1)
        command.add_argument("--no-resume", action="store_true")
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--out", type=Path, required=True)
    analysis.add_argument("--bootstrap-samples", type=int, default=10000)
    replay = sub.add_parser("replay")
    replay.add_argument("--run", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    return root


def exhaustive(config, trace, output: Path, seed, limit):
    scenario = build_scenario(config)
    validate_trace(trace, scenario)
    if scenario.mode != "static":
        raise ValueError("Exhaustive comparison requires static mode")
    count = len(scenario.vms)**len(trace)
    if count > limit:
        raise ValueError(f"{count:,} assignments exceed safety limit {limit:,}; use configs/tiny.yaml")
    snapshot, evaluator = Engine(scenario).snapshot(), Evaluator()
    best_score, best = None, None
    started = time.perf_counter()
    for vector in product(range(len(scenario.vms)), repeat=len(trace)):
        mapping = {task.id: vm for task, vm in zip(trace, vector, strict=True)}
        score = evaluator.evaluate(mapping, trace, snapshot)
        if best_score is None or score.total < best_score.total:
            best_score, best = score, mapping
    rows = [dict(algorithm="optimal", cost=best_score.total, absolute_gap=0., evaluations=count, search_s=time.perf_counter()-started)]
    for name in ALGORITHMS:
        result = Scheduler(name).solve(trace, snapshot, evaluator, scenario.budget, rng_for(seed, f"algorithm:{name}"))
        rows.append(dict(algorithm=name, cost=result.score.total, absolute_gap=result.score.total-best_score.total,
                         evaluations=result.evaluations, search_s=result.elapsed_s))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/"optimality_gaps.csv", index=False)
    (output/"optimum.json").write_text(json.dumps(dict(assignments=best, cost=best_score.total, combinations=count), indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "generate":
            if args.seed < 0:
                raise ValueError("seed must be nonnegative")
            trace = generate(build_scenario(load_config(args.config)), args.seed)
            save_trace(args.out, trace)
            print(f"Saved {len(trace)} tasks: {args.out.resolve()}")
        elif args.command == "run":
            config = load_config(args.config)
            trace = args.trace.resolve() if args.trace else prepare_trace(config, args.seed, args.out)
            summary = run_one(config, args.algorithm, args.seed, str(trace), str(args.out/"run"), not args.no_resume)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        elif args.command in ("compare", "suite"):
            configs = [load_config(args.config)] if args.command == "compare" else suite_configs(args.manifest)
            compare(configs, args.out, args.seeds, args.algorithms, args.workers, not args.no_resume)
            print(f"Analysis: {(args.out/'analysis'/'report.md').resolve()}")
        elif args.command == "analyze":
            reanalyze(args.out, args.bootstrap_samples)
        elif args.command == "replay":
            metadata = json.loads((args.run/"metadata.json").read_text(encoding="utf-8"))
            config = load_config(args.run/"config.yaml")
            source = (args.run/metadata["trace_path"]).resolve()
            from .experiments import digest
            if digest(source) != metadata["trace_sha256"]:
                raise ValueError("Saved trace checksum mismatch")
            summary = run_one(config, metadata["algorithm"], metadata["seed"], str(source), str(args.out/"run"), False)
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        elif args.command == "exhaustive":
            config = load_config(args.config)
            trace = load_trace(args.trace) if args.trace else generate(build_scenario(config), args.seed)
            exhaustive(config, trace, args.out, args.seed, args.max_combinations)
    except (ValueError, OSError, KeyError) as error:
        parser().exit(2, f"cloudedge: {error}\n")


if __name__ == "__main__":
    main()
