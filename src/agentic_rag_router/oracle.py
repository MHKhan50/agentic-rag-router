"""Oracle routing labels and data splits.

For every question we know how each agent actually did and what it cost.
The oracle label is the *cheapest agent that answered correctly*; if no agent
is correct the label falls back to the most capable (most expensive) agent and
the row is flagged, so later experiments can keep or drop those rows.

Agent cost order is measured, not assumed: agents are ranked by their mean
FLOPs over the whole run. All per-agent scores and costs are kept, so labels
can be rebuilt with another threshold or a cost-weighted utility without
re-running any model.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

import pandas as pd

from .data import Question
from .metrics import is_correct, score

COST_COLUMNS = ["llm_calls", "prompt_tokens", "gen_tokens", "retrieval_calls", "seconds", "flops"]


def score_runs(questions: List[Question], runs: Dict[str, List[dict]]) -> pd.DataFrame:
    """Long table: one row per (question, agent) with metrics and costs."""
    by_qid = {q.qid: q for q in questions}
    rows = []
    for agent, results in runs.items():
        for r in results:
            q = by_qid[r["qid"]]
            gold = set(q.gold_pids)
            row = {
                "qid": q.qid,
                "dataset": q.dataset,
                "agent": agent,
                "prediction": r["prediction"],
                **score(r["prediction"], q.answers),
                **{c: r[c] for c in COST_COLUMNS},
                "gold_recall": (len(gold & set(r["retrieved_pids"])) / len(gold)) if gold else float("nan"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def agent_cost_order(long: pd.DataFrame) -> List[str]:
    return long.groupby("agent")["flops"].mean().sort_values().index.tolist()


def build_oracle(long: pd.DataFrame, f1_threshold: float = 0.6) -> pd.DataFrame:
    """Wide table: one row per question with the oracle label."""
    order = agent_cost_order(long)
    long = add_correct(long, f1_threshold)
    correct = long.pivot(index="qid", columns="agent", values="correct")[order]
    flops = long.pivot(index="qid", columns="agent", values="flops")[order]
    meta = long.drop_duplicates("qid").set_index("qid")[["dataset"]]

    labels, none_correct = [], []
    for qid, row in correct.iterrows():
        winners = [a for a in order if row[a] == 1]
        labels.append(winners[0] if winners else order[-1])
        none_correct.append(not winners)

    wide = meta.loc[correct.index].copy()
    wide["oracle_agent"] = labels
    wide["oracle_label"] = [order.index(a) for a in labels]
    wide["none_correct"] = none_correct
    for a in order:
        wide[f"correct__{a}"] = correct[a].values
        wide[f"flops__{a}"] = flops[a].values
    wide.attrs["agent_order"] = order
    return wide.reset_index()


def assign_splits(oracle: pd.DataFrame, fractions: Dict[str, float], seed: int) -> pd.Series:
    """Stratified by source dataset so every split has the same mix."""
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    split = {}
    groups = defaultdict(list)
    for qid, ds in zip(oracle["qid"], oracle["dataset"]):
        groups[ds].append(qid)
    names = list(fractions)
    for ds, qids in sorted(groups.items()):
        qids = sorted(qids)
        random.Random(f"{seed}-{ds}-split").shuffle(qids)
        n = len(qids)
        bounds, acc = [], 0.0
        for name in names[:-1]:
            acc += fractions[name]
            bounds.append(int(acc * n + 0.5 + 1e-9))  # round half up, float-safe
        start = 0
        for name, end in zip(names, bounds + [n]):
            for qid in qids[start:end]:
                split[qid] = name
            start = end
    return oracle["qid"].map(split)


def add_correct(long: pd.DataFrame, f1_threshold: float = 0.6) -> pd.DataFrame:
    return long.assign(
        correct=[int(is_correct(r, f1_threshold)) for r in long[["em", "f1", "acc"]].to_dict("records")]
    )


def summarise(long: pd.DataFrame, oracle: pd.DataFrame, f1_threshold: float = 0.6) -> pd.DataFrame:
    """Per-agent accuracy and cost, plus the oracle router as an upper bound."""
    order = oracle.attrs.get("agent_order") or agent_cost_order(long)
    long = add_correct(long, f1_threshold)
    agg = long.groupby("agent").agg(
        correct=("correct", "mean"), em=("em", "mean"), f1=("f1", "mean"), acc=("acc", "mean"),
        llm_calls=("llm_calls", "mean"), tokens=("prompt_tokens", "mean"),
        seconds=("seconds", "mean"), gflops=("flops", lambda x: x.mean() / 1e9),
    ).loc[order]
    best = long.merge(oracle[["qid", "oracle_agent"]], left_on=["qid", "agent"], right_on=["qid", "oracle_agent"])
    agg.loc["ORACLE_ROUTER"] = {
        "correct": best["correct"].mean(), "em": best["em"].mean(), "f1": best["f1"].mean(), "acc": best["acc"].mean(),
        "llm_calls": best["llm_calls"].mean(), "tokens": best["prompt_tokens"].mean(),
        "seconds": best["seconds"].mean(), "gflops": best["flops"].mean() / 1e9,
    }
    return agg.round(4)
