from __future__ import annotations

import math
import time

import numpy as np

from .engine import Evaluator, Predictor
from .models import ScheduleResult, Score, Snapshot, Task


ALGORITHMS = ("random", "round_robin", "greedy", "ga", "aco", "sa")


class Search:
    def __init__(self, batch, snapshot, evaluator, budget):
        self.batch, self.snapshot, self.evaluator = batch, snapshot, evaluator
        self.budget = budget
        self.count = 0
        self.best: np.ndarray | None = None
        self.best_score: Score | None = None
        self.trace = []

    def evaluate(self, vector: np.ndarray) -> Score:
        if self.count >= self.budget:
            raise RuntimeError("Evaluation budget exhausted")
        mapping = {t.id: int(vm) for t, vm in zip(self.batch, vector, strict=True)}
        score = self.evaluator.evaluate(mapping, self.batch, self.snapshot)
        self.count += 1
        if self.best_score is None or score.total < self.best_score.total:
            self.best, self.best_score = vector.copy(), score
        self.trace.append(dict(evaluation=self.count, best_cost=self.best_score.total))
        return score

    def result(self, started):
        return ScheduleResult({t.id: int(vm) for t, vm in zip(self.batch, self.best, strict=True)},
                              self.best_score, self.count, time.perf_counter()-started, self.trace)


class Scheduler:
    def __init__(self, name: str):
        if name not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm {name}; choose {', '.join(ALGORITHMS)}")
        self.name = name
        self.cursor = 0

    def solve(self, batch: list[Task], snapshot: Snapshot, evaluator: Evaluator,
              budget: int, rng: np.random.Generator) -> ScheduleResult:
        started = time.perf_counter()
        if budget < 1:
            raise ValueError("Budget must be positive")
        if not batch:
            return ScheduleResult({}, Score(0., 0., 0., 0.), 0, time.perf_counter()-started, [])
        vm_count, n = len(snapshot.scenario.vms), len(batch)
        search = Search(batch, snapshot, evaluator, budget)
        if self.name == "random":
            search.evaluate(rng.integers(vm_count, size=n))
        elif self.name == "round_robin":
            search.evaluate((np.arange(n)+self.cursor) % vm_count)
            self.cursor = (self.cursor+n) % vm_count
        else:
            greedy = Predictor(snapshot).greedy(batch)
            initial = np.array([greedy[t.id] for t in batch], dtype=int)
            search.evaluate(initial)
            if self.name == "ga":
                self._ga(search, rng)
            elif self.name == "aco":
                self._aco(search, rng)
            elif self.name == "sa":
                self._sa(search, rng)
        return search.result(started)

    @staticmethod
    def _ga(search: Search, rng):
        cfg = search.snapshot.scenario.config["optimization"]["ga"]
        n, m = len(search.batch), len(search.snapshot.scenario.vms)
        population = [(search.best.copy(), search.best_score.total)]
        while len(population) < cfg["population"] and search.count < search.budget:
            vector = rng.integers(m, size=n)
            population.append((vector, search.evaluate(vector).total))
        while search.count < search.budget:
            population.sort(key=lambda pair: pair[1])
            children = population[:min(cfg["elites"], len(population))]
            # Elites do not consume evaluations, but every generation evaluates
            # at least one child even when configured elites==population.
            target = max(cfg["population"], len(children)+1)
            while len(children) < target and search.count < search.budget:
                def parent():
                    candidates = rng.integers(len(population), size=cfg["tournament"])
                    return min((population[int(i)] for i in candidates), key=lambda pair: pair[1])[0]
                vector = parent().copy()
                other = parent()
                if n > 1 and rng.random() < cfg["crossover"]:
                    point = int(rng.integers(1, n))
                    vector[point:] = other[point:]
                mask = rng.random(n) < 1/n
                vector[mask] = rng.integers(m, size=int(mask.sum()))
                children.append((vector, search.evaluate(vector).total))
            population = sorted(children, key=lambda pair: pair[1])[:cfg["population"]]

    @staticmethod
    def _aco(search: Search, rng):
        cfg = search.snapshot.scenario.config["optimization"]["aco"]
        n, m = len(search.batch), len(search.snapshot.scenario.vms)
        pheromone = np.ones((n, m))
        hints = Predictor(search.snapshot).hints(search.batch)
        while search.count < search.budget:
            iteration_best = None
            for _ in range(cfg["ants"]):
                if search.count == search.budget:
                    break
                vector = np.empty(n, dtype=int)
                additional = np.zeros(m)
                for i, task in enumerate(search.batch):
                    # Log probabilities prevent overflow from very small
                    # predicted durations or custom alpha/beta values.
                    logits = cfg["alpha"]*np.log(pheromone[i]) - cfg["beta"]*np.log(hints[i]+additional)
                    probability = np.exp(logits-logits.max())
                    probability /= probability.sum()
                    vm = int(rng.choice(m, p=probability))
                    vector[i] = vm
                    additional[vm] += task.mi / search.snapshot.scenario.vms[vm].mips / search.snapshot.scenario.vms[vm].slots
                score = search.evaluate(vector)
                if iteration_best is None or score.total < iteration_best[1]:
                    iteration_best = vector, score.total
            pheromone *= 1-cfg["rho"]
            vector, cost = iteration_best
            pheromone[np.arange(n), vector] += 1/(1+cost)
            np.clip(pheromone, 1e-6, 1e6, out=pheromone)

    @staticmethod
    def _sa(search: Search, rng):
        cfg = search.snapshot.scenario.config["optimization"]["sa"]
        n, m = len(search.batch), len(search.snapshot.scenario.vms)
        current, current_cost = search.best.copy(), search.best_score.total
        temperature = cfg["temperature"]
        steps = 0
        while search.count < search.budget:
            candidate = current.copy()
            index = int(rng.integers(n))
            # Exclude the existing target when an alternative exists.
            value = int(rng.integers(m-1)) if m > 1 else 0
            candidate[index] = value + (value >= current[index]) if m > 1 else 0
            score = search.evaluate(candidate)
            delta = score.total-current_cost
            if delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-12)):
                current, current_cost = candidate, score.total
            steps += 1
            if steps % cfg["epoch"] == 0:
                temperature *= cfg["cooling"]
