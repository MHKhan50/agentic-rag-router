"""Benchmark loading.

Four public QA sets cover the spectrum of retrieval needs a router must learn:

==========  ==============================  ===========================
key         Hugging Face id                 what it tests
==========  ==============================  ===========================
nq_open     google-research-datasets/nq_open  parametric / open-domain
squad       rajpurkar/squad                 single-hop, needs evidence
hotpotqa    hotpotqa/hotpot_qa (distractor) 2-hop reasoning
musique     dgslibisey/MuSiQue (answerable) 2-4 hop connected reasoning
==========  ==============================  ===========================

Row parsers are pure functions over plain dicts so they can be unit-tested
without network access; ``load_questions`` wraps them with ``datasets``.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Passage:
    pid: str
    title: str
    text: str

    def render(self, max_words: Optional[int] = None) -> str:
        words = self.text.split()
        if max_words is not None and len(words) > max_words:
            words = words[:max_words] + ["..."]
        body = " ".join(words)
        return f"{self.title}: {body}" if self.title else body


@dataclass
class Question:
    qid: str
    dataset: str
    question: str
    answers: List[str]
    passages: List[Passage] = field(default_factory=list)  # contributed to the shared corpus
    gold_pids: List[str] = field(default_factory=list)  # supporting passages, if annotated
    meta: Dict[str, str] = field(default_factory=dict)


def passage_id(title: str, text: str) -> str:
    """Content hash, so the same paragraph from two datasets is stored once."""
    return hashlib.sha1(f"{title}\n{text}".encode("utf-8")).hexdigest()[:16]


def _make_passage(title: str, text: str) -> Passage:
    title, text = (title or "").strip(), " ".join((text or "").split())
    return Passage(pid=passage_id(title, text), title=title, text=text)


def parse_nq_open(row: dict, idx: int) -> Question:
    return Question(
        qid=f"nq_open-{idx}",
        dataset="nq_open",
        question=row["question"].strip(),
        answers=[a for a in row["answer"] if a.strip()],
    )


def parse_squad(row: dict, idx: int) -> Question:
    p = _make_passage(row["title"].replace("_", " "), row["context"])
    answers = list(dict.fromkeys(a for a in row["answers"]["text"] if a.strip()))
    return Question(
        qid=f"squad-{row['id']}",
        dataset="squad",
        question=row["question"].strip(),
        answers=answers,
        passages=[p],
        gold_pids=[p.pid],
    )


def parse_hotpotqa(row: dict, idx: int) -> Question:
    ctx = row["context"]
    support_titles = set(row["supporting_facts"]["title"])
    passages, gold = [], []
    for title, sentences in zip(ctx["title"], ctx["sentences"]):
        p = _make_passage(title, " ".join(s.strip() for s in sentences))
        passages.append(p)
        if title in support_titles:
            gold.append(p.pid)
    return Question(
        qid=f"hotpotqa-{row['id']}",
        dataset="hotpotqa",
        question=row["question"].strip(),
        answers=[row["answer"]],
        passages=passages,
        gold_pids=gold,
        meta={"type": row.get("type", ""), "level": row.get("level", "")},
    )


def parse_musique(row: dict, idx: int) -> Optional[Question]:
    if not row.get("answerable", True):
        return None
    passages, gold = [], []
    for para in row["paragraphs"]:
        p = _make_passage(para["title"], para["paragraph_text"])
        passages.append(p)
        if para.get("is_supporting"):
            gold.append(p.pid)
    answers = list(dict.fromkeys([row["answer"], *row.get("answer_aliases", [])]))
    return Question(
        qid=f"musique-{row['id']}",
        dataset="musique",
        question=row["question"].strip(),
        answers=[a for a in answers if a.strip()],
        passages=passages,
        gold_pids=gold,
        meta={"hops": row["id"].split("__")[0]},
    )


PARSERS: Dict[str, Callable[[dict, int], Optional[Question]]] = {
    "nq_open": parse_nq_open,
    "squad": parse_squad,
    "hotpotqa": parse_hotpotqa,
    "musique": parse_musique,
}

HF_SOURCES = {
    "nq_open": ("google-research-datasets/nq_open", None, "validation"),
    "squad": ("rajpurkar/squad", None, "validation"),
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", "validation"),
    "musique": ("dgslibisey/MuSiQue", None, "validation"),
}


def sample_questions(rows: Iterable[dict], dataset: str, n: int, seed: int) -> List[Question]:
    """Parse every row, drop invalid ones, then take a seeded random sample."""
    parser = PARSERS[dataset]
    parsed = [q for i, r in enumerate(rows) if (q := parser(r, i)) is not None and q.answers]
    rng = random.Random(f"{seed}-{dataset}")
    rng.shuffle(parsed)
    return sorted(parsed[:n], key=lambda q: q.qid)


def load_questions(n_per_dataset: Dict[str, int], seed: int) -> List[Question]:
    from datasets import load_dataset  # imported lazily: heavy, network-bound

    questions: List[Question] = []
    for name, n in n_per_dataset.items():
        repo, config, split = HF_SOURCES[name]
        try:
            ds = load_dataset(repo, config, split=split)
        except Exception:
            # Hub keeps an auto-converted Parquet copy of every dataset; use it
            # if the main branch cannot be read by the installed `datasets`.
            ds = load_dataset(repo, config, split=split, revision="refs/convert/parquet")
        questions.extend(sample_questions(ds, name, n, seed))
    return questions


def build_corpus(questions: List[Question]) -> List[Passage]:
    """Pool every question's paragraphs into one shared, de-duplicated corpus.

    Retrieval is over the *whole* pool, not each question's own paragraphs, so
    agents face realistic distractors (including other datasets' passages).
    NQ-Open contributes no passages: its questions test whether an agent can
    answer from parametric knowledge, or is misled by irrelevant retrieval.
    """
    seen: Dict[str, Passage] = {}
    for q in questions:
        for p in q.passages:
            seen.setdefault(p.pid, p)
    return list(seen.values())
