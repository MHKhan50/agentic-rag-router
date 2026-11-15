"""Answer-level cascades: run cheap agents first, look at their answers, escalate only when needed.

The routers in ``router.py`` decide from the question alone, before any agent runs. A cascade instead
runs one or two cheap agents, computes signals from their answers (do two cheap agents agree? did
the small model refuse? how close is the best retrieved passage?), and sends the question to a
stronger agent only when those signals look weak. Cost is additive: every agent that ran is paid for.

All signals here are available at inference time. Gold answers and gold passages are never used as
features (``gold_recall`` in the long table is deliberately ignored).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .evaluation import fixed_mix_frontier, frontier_accuracy
from .metrics import normalize_answer, token_f1

_REFUSAL = re.compile(
    r"\b(unknown|not (?:mentioned|provided|stated|given|specified|available|enough)|cannot|can ?not|"
    r"no (?:information|answer|context)|unanswerable|i don.?t know|insufficient)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Cascade:
    """``first`` agents always run; ``answer`` (one of them) is kept unless the score is below tau,
    in which case ``escalate`` runs and its answer is used."""

    name: str
    first: Tuple[str, ...]
    answer: str
    escalate: str
    features: Tuple[str, ...] = field(default_factory=tuple)


def is_refusal(prediction: str) -> int:
    p = str(prediction or "").strip()
    return int(not p or bool(_REFUSAL.search(p)))


def n_tokens(prediction: str) -> int:
    return len(normalize_answer(str(prediction or "")).split())


def predictions_wide(long: pd.DataFrame, qids: Sequence[str], agents: Sequence[str]) -> pd.DataFrame:
    wide = long.pivot(index="qid", columns="agent", values="prediction")
    missing = [a for a in agents if a not in wide.columns]
    if missing:
        raise KeyError("agents missing from the long table: {}".format(missing))
    return wide.reindex(list(qids))[list(agents)].fillna("")


def top_retrieval_score(query_emb: np.ndarray, passage_emb: np.ndarray, pid_index: Dict[str, int],
                        retrieved: Sequence[Sequence[str]]) -> np.ndarray:
    """Cosine similarity between each query and its best retrieved passage (NaN if none)."""
    q = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    out = np.full(len(retrieved), np.nan, dtype=float)
    for i, pids in enumerate(retrieved):
        idx = [pid_index[p] for p in pids if p in pid_index]
        if idx:
            P = passage_emb[idx]
            P = P / np.linalg.norm(P, axis=1, keepdims=True)
            out[i] = float((P @ q[i]).max())
    return out


def answer_signals(preds: pd.DataFrame, retrieval: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """Per-question features from the agents' answers, named ``<signal>__<agent>`` or
    ``agree_f1__<a>__<b>`` / ``agree_em__<a>__<b>`` for every pair of agents."""
    feats: Dict[str, np.ndarray] = {}
    agents = list(preds.columns)
    for a in agents:
        col = preds[a].tolist()
        feats["len__" + a] = np.array([n_tokens(p) for p in col], dtype=float)
        feats["refusal__" + a] = np.array([is_refusal(p) for p in col], dtype=float)
    for i, a in enumerate(agents):
        for b in agents[i + 1:]:
            pa, pb = preds[a].tolist(), preds[b].tolist()
            feats["agree_f1__{}__{}".format(a, b)] = np.array([token_f1(x, y) for x, y in zip(pa, pb)])
            feats["agree_em__{}__{}".format(a, b)] = np.array(
                [float(bool(normalize_answer(x)) and normalize_answer(x) == normalize_answer(y)) for x, y in zip(pa, pb)])
    for a, scores in (retrieval or {}).items():
        feats["retr_top__" + a] = np.asarray(scores, dtype=float)
    return pd.DataFrame(feats, index=preds.index)


def cascade_outcome(c: Cascade, accept: np.ndarray, correct: np.ndarray, flops: np.ndarray,
                    order: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-question (correct, flops) for a cascade given a boolean accept mask."""
    col = {a: k for k, a in enumerate(order)}
    paid = flops[:, [col[a] for a in c.first]].sum(axis=1)
    esc_already_ran = c.escalate in c.first
    cost = paid + np.where(accept | esc_already_ran, 0.0, flops[:, col[c.escalate]])
    hit = np.where(accept, correct[:, col[c.answer]], correct[:, col[c.escalate]])
    return hit.astype(int), cost


def cascade_metrics(c: Cascade, scores: np.ndarray, tau: float, correct: np.ndarray, flops: np.ndarray,
                    order: Sequence[str]) -> Dict[str, float]:
    accept = scores >= tau
    hit, cost = cascade_outcome(c, accept, correct, flops, order)
    return {"accuracy": float(hit.mean()), "gflops": float(cost.mean() / 1e9),
            "cost_vs_strongest": float(cost.sum() / flops[:, -1].sum()),
            "escalation_rate": float(1 - accept.mean())}


def cascade_sweep(c: Cascade, scores: np.ndarray, correct: np.ndarray, flops: np.ndarray, order: Sequence[str],
                  taus: Sequence[float]) -> List[Tuple[float, float, float]]:
    pts = []
    for t in taus:
        m = cascade_metrics(c, scores, float(t), correct, flops, order)
        pts.append((float(t), m["gflops"], m["accuracy"]))
    return pts


def pick_tau(c: Cascade, val_scores: np.ndarray, correct: np.ndarray, flops: np.ndarray, order: Sequence[str],
             taus: Sequence[float], margin: float = 0.01) -> float:
    """Cheapest tau on validation within ``margin`` of the strongest agent's validation accuracy
    (same rule as notebook 02); if none reaches it, the most accurate tau."""
    target = correct[:, -1].mean() - margin
    stats = [(float(t), cascade_metrics(c, val_scores, float(t), correct, flops, order)) for t in taus]
    ok = [(t, m["gflops"]) for t, m in stats if m["accuracy"] >= target]
    if ok:
        return min(ok, key=lambda x: (x[1], x[0]))[0]
    return max(stats, key=lambda x: (x[1]["accuracy"], -x[1]["gflops"]))[0]


def fit_accept_model(F_train: np.ndarray, y_train: np.ndarray, seed: int = 0):
    """Logistic regression estimating P(answer agent is correct) from answer signals."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
    model.fit(np.nan_to_num(F_train), y_train)
    return lambda F: model.predict_proba(np.nan_to_num(F))[:, 1]


def gain_vs_fixed_mix(points: Sequence[Tuple[float, float, float]], correct: np.ndarray,
                      flops: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """(tau, gflops, accuracy, accuracy minus the fixed-agent mix at the same compute) per point."""
    hull = fixed_mix_frontier(correct, flops)
    return [(t, g, a, a - frontier_accuracy(hull, g)) for t, g, a in points]
