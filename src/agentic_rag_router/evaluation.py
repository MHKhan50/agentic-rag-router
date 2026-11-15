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


def fixed_mix_frontier(correct: np.ndarray, flops: np.ndarray) -> List[Tuple[float, float]]:
    """Upper concave hull of the fixed single-agent policies in (gflops, accuracy).

    Any point on a segment between two fixed agents is reachable without a
    router, by sending a random share of the questions to each agent. A router
    is only useful where it is above this frontier at the same compute."""
    pts = sorted((float(flops[:, k].mean() / 1e9), float(correct[:, k].mean())) for k in range(correct.shape[1]))
    hull: List[Tuple[float, float]] = []
    for p in pts:
        if hull and p[1] <= hull[-1][1]:
            continue  # dearer and not better: never on the frontier
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (p[0] - x1) <= (p[1] - y1) * (x2 - x1):
                hull.pop()  # hull[-1] lies on or below the chord: drop it
            else:
                break
        hull.append(p)
    return hull


def frontier_accuracy(hull: Sequence[Tuple[float, float]], cost: float) -> float:
    """Accuracy of the fixed-agent mix at ``cost`` (linear between hull points;
    the cheapest agent's accuracy below it, the best agent's above it)."""
    if cost <= hull[0][0]:
        return float(hull[0][1])
    for (x1, y1), (x2, y2) in zip(hull, hull[1:]):
        if cost <= x2:
            return float(y1 + (y2 - y1) * (cost - x1) / (x2 - x1))
    return float(hull[-1][1])


def area_on_range(points: Sequence[Tuple[float, float]], lo: float, hi: float) -> float:
    """Mean accuracy of a step Pareto front over the compute range [lo, hi].

    Below the front's cheapest point the policy is credited with the cheapest
    point's accuracy only if that point is inside the range; the range should be
    chosen so that every compared curve covers it."""
    front = pareto_front(points)
    grid = np.linspace(lo, hi, 401)
    vals = []
    for c in grid:
        ok = [a for cost, a in front if cost <= c + 1e-9]
        vals.append(max(ok) if ok else np.nan)
    return float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")


def frontier_area_on_range(hull: Sequence[Tuple[float, float]], lo: float, hi: float) -> float:
    """Mean accuracy of the fixed-agent mix over [lo, hi]."""
    grid = np.linspace(lo, hi, 401)
    return float(np.mean([frontier_accuracy(hull, c) for c in grid]))
