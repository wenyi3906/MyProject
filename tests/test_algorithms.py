from itertools import product

import numpy as np
import pytest

from cloudedge.algorithms import ALGORITHMS, Scheduler
from cloudedge.engine import Engine, Evaluator, Predictor
from cloudedge.workload import rng_for
from conftest import scenario, task


@pytest.mark.parametrize("name", ALGORITHMS)
@pytest.mark.parametrize("n,budget", [(1, 1), (1, 15), (4, 40)])
def test_budget_validity_reproducibility_and_optimality_gap(name, n, budget):
    s = scenario(topology={"vms_per_edge": 2, "cloud_mips": 200})
    batch = [task(i, mi=50+i*20, delay=2) for i in range(n)]
    snapshot = Engine(s).snapshot()
    evaluator = Evaluator()
    result = Scheduler(name).solve(batch, snapshot, evaluator, budget, rng_for(7, name))
    again = Scheduler(name).solve(batch, snapshot, evaluator, budget, rng_for(7, name))
    assert result.assignments == again.assignments
    assert result.score == again.score
    assert result.convergence == again.convergence
    assert result.evaluations == (budget if name in ("ga", "aco", "sa") else 1)
    assert set(result.assignments) == {t.id for t in batch}
    assert result.score == evaluator.evaluate(result.assignments, batch, snapshot)
    optimum = min(evaluator.evaluate({t.id: v for t, v in zip(batch, vector)}, batch, snapshot).total
                  for vector in product(range(3), repeat=n))
    assert result.score.total >= optimum-1e-12
    if name in ("ga", "aco", "sa"):
        greedy = evaluator.evaluate(Predictor(snapshot).greedy(batch), batch, snapshot)
        assert result.score.total <= greedy.total+1e-12
    costs = [row["best_cost"] for row in result.convergence]
    assert all(a >= b for a, b in zip(costs, costs[1:]))


def test_three_component_cost_is_normalized_and_minimized():
    s = scenario(topology={"vms_per_edge": 2}, optimization={"weights": [.5, .3, .2]})
    batch = [task(0, delay=.5), task(1, delay=2)]
    score = Evaluator().evaluate({0: 0, 1: 1}, batch, Engine(s).snapshot())
    assert score.failure_rate == .5
    assert 0 <= score.normalized_delay <= 1
    assert 0 <= score.imbalance <= 1
    assert score.total == pytest.approx(.5*score.failure_rate+.3*score.normalized_delay+.2*score.imbalance)
