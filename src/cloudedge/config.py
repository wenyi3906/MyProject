from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
from typing import Any

import yaml

from .models import Link, Scenario, VM


DEFAULT_APPS = [
    dict(name="ar", share=.3, interval=5, max_delay=.5, input_kb=1500, output_kb=25, mi=2000),
    dict(name="health", share=.2, interval=20, max_delay=5, input_kb=5000, output_kb=20, mi=800),
    dict(name="heavy", share=.2, interval=60, max_delay=10, input_kb=12500, output_kb=250, mi=15000),
    dict(name="info", share=.3, interval=20, max_delay=2, input_kb=25, output_kb=2000, mi=50),
]
DEFAULT_CONFIG: dict[str, Any] = {
    "name": "dynamic_base", "mode": "dynamic",
    "simulation": {"duration": 300., "warmup": 30., "window": .1, "max_batch": 128, "sample_interval": 1.},
    "topology": {"aps": 8, "edge_hosts": 8, "vms_per_edge": 4, "edge_mips": 10000., "cloud_vms": 4, "cloud_mips": 100000., "slots_per_vm": 1},
    "network": {"wlan_mbps": 200., "wan_mbps": 1000., "wlan_propagation": .005, "wan_propagation": .05},
    "workload": {"devices": 1000, "task_count": 500, "release_window": 30., "apps": DEFAULT_APPS, "bursts": []},
    "optimization": {"weights": [.5, .3, .2], "static_budget": 2000, "dynamic_budget": 200,
        "ga": {"population": 30, "tournament": 3, "crossover": .8, "elites": 2},
        "aco": {"ants": 20, "alpha": 1., "beta": 2., "rho": .2},
        "sa": {"temperature": .1, "cooling": .95, "epoch": 20}},
    "experiment": {"algorithms": ["random", "round_robin", "greedy", "ga", "aco", "sa"], "seeds": list(range(100, 130)), "bootstrap_samples": 10000},
}


def merge(base: dict, changes: dict) -> dict:
    out = deepcopy(base)
    for key, value in changes.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: str | Path, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Circular configuration extends: {path}")
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    parent = raw.pop("extends", None)
    base = load_config(path.parent / parent, _seen | {path}) if parent else DEFAULT_CONFIG
    config = merge(base, raw)
    build_scenario(config)
    return config


def _positive(value: Any, label: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"{label} must be {'nonnegative' if allow_zero else 'positive'} and finite")
    return number


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    number = _positive(value, label)
    if number != int(number):
        raise ValueError(f"{label} must be an integer")
    return int(number)


def build_scenario(config: dict[str, Any]) -> Scenario:
    cfg = merge(DEFAULT_CONFIG, config)
    def check_keys(value, template, prefix=""):
        if not isinstance(value, dict):
            raise ValueError(f"{prefix or 'configuration'} must be a mapping")
        for key, item in value.items():
            if not prefix and key == "_scan":
                continue
            if key not in template:
                raise ValueError(f"Unknown configuration key: {prefix}{key}")
            if isinstance(template[key], dict):
                check_keys(item, template[key], f"{prefix}{key}.")
    check_keys(cfg, DEFAULT_CONFIG)
    name = cfg["name"]
    if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Scenario name must use ASCII letters, digits, '_' and '-'")
    sim, top, net, opt = (cfg[k] for k in ("simulation", "topology", "network", "optimization"))
    mode = cfg["mode"]
    if mode not in ("static", "dynamic"):
        raise ValueError("mode must be static or dynamic")
    duration = _positive(sim["duration"], "duration")
    warmup = _positive(sim["warmup"], "warmup", True)
    if warmup >= duration or (mode == "static" and warmup != 0):
        raise ValueError("Require warmup < duration; static mode requires warmup=0")
    if not isinstance(opt["weights"], (list, tuple)):
        raise ValueError("weights must be a three-element list")
    weights = tuple(_positive(x, "objective weight", True) for x in opt["weights"])
    if len(weights) != 3 or any(not math.isfinite(x) or x < 0 for x in weights) or not math.isclose(sum(weights), 1, abs_tol=1e-10):
        raise ValueError("Three nonnegative objective weights must sum to 1")
    aps = _count(top["aps"], "aps")
    slots = _count(top["slots_per_vm"], "slots_per_vm")
    vms = []
    edge_hosts, per_host = _count(top["edge_hosts"], "edge_hosts"), _count(top["vms_per_edge"], "vms_per_edge")
    edge_count = edge_hosts * per_host
    edge_mips = top["edge_mips"]
    if isinstance(edge_mips, list):
        if len(edge_mips) != edge_count:
            raise ValueError("edge_mips list must contain one entry per edge VM")
        speeds = edge_mips
    else:
        speeds = [edge_mips] * edge_count
    for i, speed in enumerate(speeds):
        vms.append(VM(i, f"edge_{i // per_host}", "edge", _positive(speed, "edge_mips"), slots))
    for _ in range(_count(top["cloud_vms"], "cloud_vms")):
        vms.append(VM(len(vms), "cloud_0", "cloud", _positive(top["cloud_mips"], "cloud_mips"), slots))
    wlan = _positive(net["wlan_mbps"], "wlan_mbps")
    wan = _positive(net["wan_mbps"], "wan_mbps")
    wlan_prop = _positive(net["wlan_propagation"], "wlan_propagation", True)
    wan_prop = _positive(net["wan_propagation"], "wan_propagation", True)
    links = tuple([Link(f"wlan_{ap}_{direction}", wlan, wlan_prop) for ap in range(aps) for direction in ("up", "down")]
                  + [Link(f"wan_{direction}", wan, wan_prop) for direction in ("up", "down")])
    apps = cfg["workload"]["apps"]
    if not apps or len({a["name"] for a in apps}) != len(apps):
        raise ValueError("Application names must be unique and nonempty")
    if not math.isclose(sum(float(a["share"]) for a in apps), 1, abs_tol=1e-10):
        raise ValueError("Application shares must sum to 1")
    for app in apps:
        if set(app) != set(DEFAULT_APPS[0]):
            raise ValueError("Each app must contain name/share/interval/max_delay/input_kb/output_kb/mi")
        for key in ("share", "interval", "max_delay", "mi"):
            _positive(app[key], f"{app['name']}.{key}", key == "share")
        for key in ("input_kb", "output_kb"):
            _positive(app[key], f"{app['name']}.{key}", True)
    _count(cfg["workload"]["devices"], "devices")
    _count(cfg["workload"]["task_count"], "task_count")
    _positive(cfg["workload"]["release_window"], "release_window", True)
    bursts = sorted(cfg["workload"]["bursts"], key=lambda b: b["start"])
    previous_end = 0.
    for burst in bursts:
        start, end = float(burst["start"]), float(burst["end"])
        if not 0 <= start < end <= duration or start < previous_end:
            raise ValueError("Burst intervals must be nonoverlapping and inside simulation duration")
        _positive(burst["multiplier"], "burst multiplier")
        previous_end = end
    for key in ("static_budget", "dynamic_budget"):
        _count(opt[key], key)
    experiment = cfg["experiment"]
    _count(experiment["bootstrap_samples"], "bootstrap_samples")
    if not experiment["seeds"] or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in experiment["seeds"]):
        raise ValueError("experiment.seeds must contain nonnegative integers")
    if len(set(experiment["seeds"])) != len(experiment["seeds"]):
        raise ValueError("experiment.seeds must be unique")
    choices = {"random", "round_robin", "greedy", "ga", "aco", "sa"}
    algorithms = experiment["algorithms"]
    if not algorithms or any(a not in choices for a in algorithms) or len(set(algorithms)) != len(algorithms):
        raise ValueError("experiment.algorithms must be unique known algorithms")
    ga, aco, sa = opt["ga"], opt["aco"], opt["sa"]
    population = _count(ga["population"], "population")
    elites = _count(ga["elites"], "elites")
    if elites > population:
        raise ValueError("elites must not exceed population")
    _count(ga["tournament"], "tournament")
    if not 0 <= float(ga["crossover"]) <= 1:
        raise ValueError("crossover must be in [0,1]")
    _positive(ga["crossover"], "crossover", True)
    _count(aco["ants"], "ants")
    _positive(aco["alpha"], "alpha", True)
    _positive(aco["beta"], "beta", True)
    if not 0 < float(aco["rho"]) < 1:
        raise ValueError("rho must be in (0,1)")
    _positive(aco["rho"], "rho")
    _positive(sa["temperature"], "temperature")
    if not 0 < float(sa["cooling"]) <= 1:
        raise ValueError("cooling must be in (0,1]")
    _positive(sa["cooling"], "cooling")
    _count(sa["epoch"], "epoch")
    return Scenario(str(cfg["name"]), mode, duration, warmup, _positive(sim["window"], "window"),
        _count(sim["max_batch"], "max_batch"), _positive(sim["sample_interval"], "sample_interval"),
        int(opt[f"{mode}_budget"]), weights, tuple(vms), links, aps, cfg)

