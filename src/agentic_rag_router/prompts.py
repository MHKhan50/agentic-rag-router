"""Prompt templates shared by all agents.

Every agent is asked for the same output format (``Answer: <short answer>``)
so that differences in accuracy come from the model and retrieval strategy,
not from prompt wording.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

from .data import Passage
from .llm import Messages

SYSTEM = (
    "You are a precise question-answering assistant. "
    "Give the shortest correct answer (an entity, number, date or short phrase), "
    "with no explanation."
)


def _context(passages: List[Passage], max_words: int) -> str:
    return "\n".join(f"[{i + 1}] {p.render(max_words)}" for i, p in enumerate(passages))


def direct_prompt(question: str) -> Messages:
    user = f"Question: {question}\nReply in the form 'Answer: <short answer>'."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def rag_prompt(question: str, passages: List[Passage], max_words: int) -> Messages:
    user = (
        f"Passages:\n{_context(passages, max_words)}\n\n"
        f"Question: {question}\n"
        "Use the passages if they contain the answer. If they do not, answer from your own "
        "knowledge; never reply that the answer is not mentioned. "
        "Reply in the form 'Answer: <short answer>'."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def step_prompt(
    question: str, passages: List[Passage], max_words: int, previous_queries: Sequence[str] = ()
) -> Messages:
    tried = ""
    if previous_queries:
        tried = "Searches already made (do not repeat them): " + "; ".join(previous_queries) + "\n"
    user = (
        f"Passages:\n{_context(passages, max_words)}\n\n"
        f"Question: {question}\n{tried}"
        "If the passages give enough information to answer, reply exactly "
        "'Answer: <short answer>'.\n"
        "If a fact is still missing, reply exactly 'Search: <one search query for the missing fact>'.\n"
        "Reply with one line only."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


_ANSWER = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE)
_SEARCH = re.compile(r"search\s*:\s*(.+)", re.IGNORECASE)


_PREFIX = re.compile(r"^\s*answer\s*:\s*", re.IGNORECASE)


def parse_answer(text: str) -> str:
    """Extract the short answer; tolerate a missing or repeated "Answer:" prefix."""
    m = _ANSWER.search(text)
    ans = m.group(1) if m else text
    ans = ans.strip().splitlines()[0] if ans.strip() else ""
    while _PREFIX.match(ans):
        ans = _PREFIX.sub("", ans, count=1)
    return ans.strip().strip("\"'").rstrip(".").strip()


def parse_step(text: str) -> Tuple[str, Optional[str]]:
    """Return ("answer", answer) or ("search", query) for an iterative step.

    Also catches "Answer: Search: ..." - a search request wrapped in the
    answer prefix - which models occasionally produce.
    """
    first = text.strip().splitlines()[0] if text.strip() else ""
    while _PREFIX.match(first):
        first = _PREFIX.sub("", first, count=1)
    s = _SEARCH.match(first.strip())
    if s and s.group(1).strip():
        return "search", s.group(1).strip()
    return "answer", parse_answer(text)
