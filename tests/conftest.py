from cloudedge.config import build_scenario, merge, DEFAULT_CONFIG
from cloudedge.models import Task


def scenario(**changes):
    base = merge(DEFAULT_CONFIG, {
        "name": "test", "mode": "static",
        "simulation": {"duration": 5, "warmup": 0, "window": .1, "sample_interval": .1},
        "topology": {"aps": 1, "edge_hosts": 1, "vms_per_edge": 1, "edge_mips": 100, "cloud_vms": 1, "cloud_mips": 1000},
        "network": {"wlan_mbps": 1, "wan_mbps": 2, "wlan_propagation": 0, "wan_propagation": 0},
        "optimization": {"static_budget": 20, "dynamic_budget": 10},
        "experiment": {"seeds": [0, 1], "bootstrap_samples": 100},
    })
    return build_scenario(merge(base, changes))


def task(id=0, release=0., mi=100., delay=10., up=0, down=0):
    return Task(id, release, 0, "test", mi, up, down, delay)
