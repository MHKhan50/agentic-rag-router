"""Tests for answer-level cascades (CPU, no downloads)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rag_router import cascade as cs  # noqa: E402

ORDER = ["A", "C", "E"]


def _long() -> pd.DataFrame:
    rows = []
    preds = {"q1": ("Paris", "paris", "Paris"), "q2": ("Rome", "Milan", "Milan"),
             "q3": ("", "unknown", "Oslo"), "q4": ("1912", "the 1912", "1913")}
    for q, ps in preds.items():
        for a, p in zip(ORDER, ps):
            rows.append({"qid": q, "agent": a, "prediction": p})
    return pd.DataFrame(rows)


def test_signals_agreement_refusal_and_length():
    preds = cs.predictions_wide(_long(), ["q1", "q2", "q3", "q4"], ORDER)
    F = cs.answer_signals(preds, {"C": np.array([0.9, 0.5, np.nan, 0.7])})
    assert list(F["agree_em__A__C"]) == [1.0, 0.0, 0.0, 1.0]   # "the 1912" normalises to "1912"
    assert F.loc["q2", "agree_f1__A__C"] == 0.0
    assert list(F["refusal__C"]) == [0, 0, 1, 0]
    assert F.loc["q3", "refusal__A"] == 1                       # empty answer counts as refusal
    assert F.loc["q4", "len__C"] == 1
    assert np.isnan(F.loc["q3", "retr_top__C"])


def test_missing_agent_raises():
    with pytest.raises(KeyError):
        cs.predictions_wide(_long(), ["q1"], ["A", "Z"])


def test_cascade_cost_is_additive_and_accuracy_switches():
    correct = np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0]])
    flops = np.array([[1e9, 2e9, 10e9]] * 4)
    c = cs.Cascade("A+C then E", ("A", "C"), "C", "E")
    accept = np.array([True, False, False, True])
    hit, cost = cs.cascade_outcome(c, accept, correct, flops, ORDER)
    assert hit.tolist() == [1, 1, 1, 1]
    assert (cost / 1e9).tolist() == [3, 13, 13, 3]
    m = cs.cascade_metrics(c, accept.astype(float), 0.5, correct, flops, ORDER)
    assert m == {"accuracy": 1.0, "gflops": 8.0, "cost_vs_strongest": 0.8, "escalation_rate": 0.5}
    # escalating to an agent that already ran costs nothing extra
    c2 = cs.Cascade("A+C, keep C else A", ("A", "C"), "C", "A")
    _, cost2 = cs.cascade_outcome(c2, np.array([False] * 4), correct, flops, ORDER)
    assert (cost2 / 1e9).tolist() == [3, 3, 3, 3]


def test_sweep_ends_and_pick_tau():
    correct = np.array([[0, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1]])
    flops = np.array([[1e9, 2e9, 10e9]] * 4)
    c = cs.Cascade("C then E", ("C",), "C", "E")
    scores = np.array([0.9, 0.2, 0.8, 0.1])                     # high exactly where C is right
    pts = cs.cascade_sweep(c, scores, correct, flops, ORDER, [0.0, 0.5, 1.01])
    assert pts[0] == (0.0, 2.0, 0.5)                             # accept all: C only
    assert pts[1] == (0.5, 7.0, 1.0)                             # perfect split
    assert pts[2] == (1.01, 12.0, 1.0)                           # escalate all: C + E
    assert cs.pick_tau(c, scores, correct, flops, ORDER, [0.0, 0.5, 1.01]) == 0.5
    g = cs.gain_vs_fixed_mix(pts, correct, flops)
    assert g[1][3] > 0                                           # beats the fixed mix at 7 GFLOPs


def test_retrieval_score_and_accept_model():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    P = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 2.0]])
    s = cs.top_retrieval_score(q, P, {"p0": 0, "p1": 1, "p2": 2}, [["p1", "p0"], ["zz"]])
    assert s[0] == pytest.approx(1.0) and np.isnan(s[1])
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    F = rng.normal(size=(200, 2))
    y = (F[:, 0] > 0).astype(int)
    f = cs.fit_accept_model(F, y)
    p = f(F)
    assert p.shape == (200,) and ((p > 0.5) == y).mean() > 0.9
