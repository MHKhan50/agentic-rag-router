"""Answer scoring with the standard SQuAD normalisation (Rajpurkar et al., 2016)."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)
_NUMBER_WORDS = {
    w: str(i)
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}


def normalize_answer(s: str) -> str:
    """SQuAD normalisation, plus number words zero-twenty mapped to digits
    so that "three" and "3" match."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(_NUMBER_WORDS.get(t, t) for t in s.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def contains(prediction: str, gold: str) -> float:
    """Normalised gold answer appears inside the prediction ("Acc" in Adaptive-RAG)."""
    g = normalize_answer(gold)
    return float(bool(g) and f" {g} " in f" {normalize_answer(prediction)} ")


def score(prediction: str, golds: List[str]) -> Dict[str, float]:
    """Max over gold aliases, as in the official evaluation scripts."""
    return {
        "em": max(exact_match(prediction, g) for g in golds),
        "f1": max(token_f1(prediction, g) for g in golds),
        "acc": max(contains(prediction, g) for g in golds),
    }


def is_correct(scores: Dict[str, float], f1_threshold: float = 0.6) -> bool:
    """Exact match, gold answer contained in a short prediction, or high token F1.

    The F1 threshold sits above 0.5 on purpose: at exactly 0.5 a prediction
    shares only half its tokens with the gold answer ("Articles 101(3) and
    102" vs "Articles 106 and 107"), which is wrong far more often than right.
    """
    return bool(scores["em"] == 1 or scores["acc"] == 1 or scores["f1"] >= f1_threshold)
