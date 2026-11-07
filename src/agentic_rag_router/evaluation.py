"""Evaluation of routing policies (NumPy only).

A *policy* is an array ``choice`` with one agent index per question. Given
the per-question correctness matrix ``correct`` (n x k, 0/1) and cost matrix
``flops`` (n x k) measured in stage 1, every policy gets the same three
numbers: answer accuracy, mean compute, and agreement with the oracle label.
"""

from __future__ import annotations

from math import comb
from typing import Dict, List, Sequence, Tuple

import numpy as np


def policy_metrics(
    choice: np.ndarray,
    correct: np.ndarray,
    flops: np.ndarray,
    oracle_label: np.ndarray | None = None,
) -> Dict[str, float]:
    rows = np.arange(len(choice))
    hit = correct[rows, choice]
    cost = flops[rows, choice]
    out = {
        "accuracy": float(hit.mean()),
        "gflops": float(cost.mean() / 1e9),
        "cost_vs_strongest": float(cost.sum() / flops[:, -1].sum()),
    }
    if oracle_label is not None:
        out["routing_accuracy"] = float((choice == oracle_label).mean())
    out["share_per_agent"] = np.bincount(choice, minlength=correct.shape[1]).astype(float).tolist()
    return out


def confidence_escalation(probs: np.ndarray, tau: float) -> np.ndarray:
    """Take the router's top agent unless its probability is below ``tau``;
    then escalate to the strongest agent (last column)."""
    choice = probs.argmax(axis=1)
    choice[probs.max(axis=1) < tau] = probs.shape[1] - 1
    return choice


def cheapest_confident(p_correct: np.ndarray, tau: float) -> np.ndarray:
    """For per-agent correctness estimates: the cheapest agent whose estimated
    probability of answering correctly is at least ``tau`` (else strongest)."""
    ok = p_correct >= tau
    choice = np.where(ok.any(axis=1), ok.argmax(axis=1), p_correct.shape[1] - 1)
    return choice.astype(int)


def sweep(policy_fn, scores: np.ndarray, correct: np.ndarray, flops: np.ndarray,
          taus: Sequence[float] = tuple(np.linspace(0, 1, 41))) -> List[Tuple[float, float, float]]:
    """(tau, gflops, accuracy) for each threshold: one accuracy-cost curve."""
    pts = []
    for t in taus:
        m = policy_metrics(policy_fn(scores, float(t)), correct, flops)
        pts.append((float(t), m["gflops"], m["accuracy"]))
    return pts


def pareto_front(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Points (cost, accuracy) not dominated by a cheaper-and-better point."""
    front, best = [], -np.inf
    for cost, acc in sorted(points, key=lambda p: (p[0], -p[1])):
        if acc > best:
            front.append((cost, acc))
            best = acc
    return front


def area_under_curve(points: Sequence[Tuple[float, float]], max_cost: float) -> float:
    """Normalised area under the Pareto front over [0, max_cost] (step
    interpolation): a single number to compare cost-accuracy trade-offs."""
    front = pareto_front(points)
    area, prev_cost, prev_acc = 0.0, 0.0, 0.0
    for cost, acc in front:
        c = min(cost, max_cost)
        area += (c - prev_cost) * prev_acc
        prev_cost, prev_acc = c, acc
    area += (max_cost - prev_cost) * prev_acc
    return float(area / max_cost)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> Dict[str, float]:
    """Exact (binomial) McNemar test on paired 0/1 outcomes."""
    b = int(((a_correct == 1) & (b_correct == 0)).sum())
    c = int(((a_correct == 0) & (b_correct == 1)).sum())
    n = b + c
    if n == 0:
        return {"a_only": b, "b_only": c, "p_value": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return {"a_only": b, "b_only": c, "p_value": float(p)}


def expected_calibration_error(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-width bins (Guo et al., 2017)."""
    probs, outcomes = probs.ravel(), outcomes.ravel().astype(float)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(probs[m].mean() - outcomes[m].mean())
    return float(ece)
