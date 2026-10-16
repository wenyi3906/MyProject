from copy import deepcopy
from dataclasses import asdict

import pytest

from cloudedge.engine import Engine, Evaluator
from conftest import scenario, task


@pytest.mark.parametrize("vm,expected", [(0, 2.7), (1, 2.95)])
def test_hand_calculated_edge_and_cloud_paths(vm, expected):
    s = scenario(network={"wlan_propagation": .1, "wan_propagation": .2})
    t = task(up=125000, down=62500)
    engine = Engine(s)
    engine.release_assigned(t, vm)
    engine.advance(t.deadline)
    row = engine.task_rows()[0]
    assert row["status"] == "success"
    assert row["latency_s"] == pytest.approx(expected)
    assert sum(v for k, v in row.items() if k.endswith(("_queue_s", "_service_s", "_propagation_s"))) == pytest.approx(expected)


@pytest.mark.parametrize("mi,delay,status", [(100, 1., "success"), (100, .999, "failed"), (30, .3, "success")])
def test_completion_at_deadline_precedes_timeout(mi, delay, status):
    engine = Engine(scenario())
    engine.release_assigned(task(mi=mi, delay=delay), 0)
    engine.advance(2.)
    assert engine.tasks[0].status == status
    assert not engine.resources["vm_0"].active


def test_timeout_running_task_releases_capacity_immediately():
    engine = Engine(scenario())
    engine.release_assigned(task(0, delay=.5), 0)
    engine.release_assigned(task(1, delay=2.), 0)
    engine.advance(2.)
    first, second = engine.task_rows()
    assert first["failure_reason"] == "compute"
    assert first["compute_service_s"] == .5
    assert second["compute_started"] == .5
    assert second["finished"] == 1.5
    assert sum(b-a for a, b in engine.resources["vm_0"].intervals) == 1.5


def test_timeout_waiting_task_removed_without_disturbing_fifo():
    engine = Engine(scenario())
    for t in (task(0, delay=2), task(1, delay=.25), task(2, delay=3)):
        engine.release_assigned(t, 0)
    engine.advance(3)
    rows = engine.task_rows()
    assert rows[1]["failure_reason"] == "compute_queue"
    assert rows[1]["compute_queue_s"] == .25
    assert rows[2]["compute_started"] == 1.
    assert rows[2]["finished"] == 2.
    assert not engine.events and not engine.live and not engine.pending


def test_propagation_does_not_hold_bandwidth():
    engine = Engine(scenario(network={"wlan_propagation": 1}))
    for t in (task(0, mi=1, up=125000), task(1, mi=1, up=125000)):
        engine.release_assigned(t, 0)
    engine.advance(10)
    rows = engine.task_rows()
    assert rows[0]["wlan_upload_service_end"] == 1.
    assert rows[0]["wlan_upload_finished"] == 2.
    assert rows[1]["wlan_upload_started"] == 1.


def test_upload_and_download_have_independent_queues():
    engine = Engine(scenario())
    engine.release_assigned(task(0, mi=1, down=125000), 0)
    engine.release_assigned(task(1, release=.1, mi=1, up=125000), 0)
    engine.advance(10)
    first, second = engine.task_rows()
    assert first["wlan_download_started"] == .01
    assert second["wlan_upload_started"] == .1


def test_compute_parallelism_and_slot_utilization():
    engine = Engine(scenario(topology={"slots_per_vm": 2}))
    for i in range(3):
        engine.release_assigned(task(i), 0)
    engine.advance(2)
    assert [rt.finished for rt in engine.tasks.values()] == [1., 1., 2.]
    assert engine.utilization("vm_0", 0, 2) == .75


@pytest.mark.parametrize("at", [0., .2, 1.05, 1.2])
def test_snapshot_replay_matches_inflight_queue_execution(at):
    s = scenario(network={"wlan_propagation": .1, "wan_propagation": .2})
    engine = Engine(s)
    first = task(0, mi=80, up=125000, down=62500)
    engine.release_assigned(first, 1)
    engine.advance(at)
    second = task(1, release=at, mi=100, delay=4, up=62500, down=25000)
    engine.add_task(second)
    snapshot = engine.snapshot()
    before = deepcopy(snapshot)
    predicted = Evaluator().evaluate({1: 0}, [second], snapshot)
    assert snapshot == before
    actual = Engine.from_snapshot(snapshot)
    actual.dispatch(1, 0)
    actual.advance(second.deadline)
    assert predicted == actual.score([second], at, second.deadline)
    assert engine.tasks[1].status == "pending"


def test_snapshot_removes_past_history_and_future_trace():
    engine = Engine(scenario())
    engine.release_assigned(task(0, delay=2), 0)
    engine.release_assigned(task(1, release=4), 1)
    engine.advance(2)
    snapshot = engine.snapshot()
    assert not snapshot.tasks
    assert not snapshot.events
    assert all(not r.intervals for r in snapshot.resources.values())


def test_dynamic_evaluator_rejects_future_and_invalid_assignments():
    snap = Engine(scenario(mode="dynamic")).snapshot()
    evaluator = Evaluator()
    with pytest.raises(ValueError, match="future"):
        evaluator.evaluate({0: 0}, [task(release=1)], snap)
    with pytest.raises(ValueError, match="exactly"):
        evaluator.evaluate({}, [task()], snap)
    with pytest.raises(ValueError, match="Invalid VM"):
        evaluator.evaluate({0: 20}, [task()], snap)


def test_timeout_during_propagation_releases_no_extra_capacity():
    engine = Engine(scenario(network={"wlan_propagation": 1}))
    engine.release_assigned(task(0, mi=1, up=125000, delay=1.5), 0)
    engine.advance(2)
    row = engine.task_rows()[0]
    assert row["failure_reason"] == "wlan_upload_propagation"
    assert row["wlan_upload_service_s"] == 1
    assert row["wlan_upload_propagation_s"] == .5
    assert not engine.resources["wlan_0_up"].active
