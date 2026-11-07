"""The four specialist agents the router chooses between.

Each agent answers a batch of questions and reports, per question, the
answer plus the exact resources it consumed (LLM calls, prompt/generated
tokens, retrieval calls, wall time). Those measurements become the cost side
of the router's accuracy/cost trade-off.

Strategies:
  * ``direct``        - no retrieval; parametric knowledge only.
  * ``single_rag``    - retrieve once with the question, answer once.
  * ``iterative_rag`` - interleave reasoning and retrieval: at each step the
    model either answers or issues a follow-up search query (in the spirit of
    IRCoT, Trivedi et al., 2023); the last step is forced to answer.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List

import numpy as np

from . import prompts
from .config import AgentSpec
from .data import Passage, Question
from .llm import ChatLLM, Generation
from .metrics import normalize_answer
from .retrieval import DenseRetriever

EncodeFn = Callable[[List[str]], np.ndarray]


@dataclass
class AgentResult:
    qid: str
    agent: str
    prediction: str
    llm_calls: int = 0
    prompt_tokens: int = 0
    gen_tokens: int = 0
    retrieval_calls: int = 0
    seconds: float = 0.0
    flops: float = 0.0  # ~2 * params * tokens processed (Kaplan et al., 2020)
    retrieved_pids: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)  # raw step outputs, for auditing

    def to_dict(self) -> dict:
        return asdict(self)

    def add(self, g: Generation, n_params: int) -> None:
        self.llm_calls += 1
        self.prompt_tokens += g.prompt_tokens
        self.gen_tokens += g.gen_tokens
        self.seconds += g.seconds
        self.flops += 2.0 * n_params * (g.prompt_tokens + g.gen_tokens)


class Agent:
    def __init__(
        self,
        spec: AgentSpec,
        llm: ChatLLM,
        retriever: DenseRetriever | None = None,
        encode_queries: EncodeFn | None = None,
        passage_max_words: int = 120,
        answer_max_new_tokens: int = 32,
        step_max_new_tokens: int = 48,
    ):
        if spec.strategy != "direct" and (retriever is None or encode_queries is None):
            raise ValueError(f"{spec.name}: retrieval strategy needs a retriever and encoder")
        self.spec, self.llm, self.retriever, self.encode = spec, llm, retriever, encode_queries
        self.max_words = passage_max_words
        self.answer_tokens, self.step_tokens = answer_max_new_tokens, step_max_new_tokens

    def run(self, questions: List[Question]) -> List[AgentResult]:
        strategy = {
            "direct": self._direct,
            "single_rag": self._single_rag,
            "iterative_rag": self._iterative,
        }[self.spec.strategy]
        return strategy(questions)

    # -- helpers --------------------------------------------------------
    def _new(self, q: Question) -> AgentResult:
        return AgentResult(qid=q.qid, agent=self.spec.name, prediction="")

    def _retrieve(self, queries: List[str], results: List[AgentResult]) -> List[List[Passage]]:
        t0 = time.perf_counter()
        hits = self.retriever.search(self.encode(queries), self.spec.top_k)
        per_item = (time.perf_counter() - t0) / max(len(queries), 1)
        for r in results:
            r.retrieval_calls += 1
            r.seconds += per_item
        return hits

    def _generate(self, prompts_: List[list], max_new: int) -> List[Generation]:
        return self.llm.generate(prompts_, max_new_tokens=max_new, batch_size=self.spec.batch_size)

    # -- strategies -----------------------------------------------------
    def _direct(self, questions: List[Question]) -> List[AgentResult]:
        results = [self._new(q) for q in questions]
        gens = self._generate([prompts.direct_prompt(q.question) for q in questions], self.answer_tokens)
        for r, g in zip(results, gens):
            r.add(g, self.llm.n_params)
            r.trace.append(g.text)
            r.prediction = prompts.parse_answer(g.text)
        return results

    def _single_rag(self, questions: List[Question]) -> List[AgentResult]:
        results = [self._new(q) for q in questions]
        hits = self._retrieve([q.question for q in questions], results)
        ctx = [h[: self.spec.max_passages] for h in hits]
        gens = self._generate(
            [prompts.rag_prompt(q.question, c, self.max_words) for q, c in zip(questions, ctx)],
            self.answer_tokens,
        )
        for r, c, g in zip(results, ctx, gens):
            r.add(g, self.llm.n_params)
            r.retrieved_pids = [p.pid for p in c]
            r.trace.append(g.text)
            r.prediction = prompts.parse_answer(g.text)
        return results

    def _iterative(self, questions: List[Question]) -> List[AgentResult]:
        """Up to ``max_steps`` rounds. A round either answers or searches; a
        repeated search query brings no new evidence, so that question is
        made to answer in its next round instead of searching again."""
        results = [self._new(q) for q in questions]
        hits = self._retrieve([q.question for q in questions], results)
        ctx: Dict[int, List[Passage]] = {i: list(h) for i, h in enumerate(hits)}
        # The question itself was the first search, so it counts as already tried.
        queries: Dict[int, List[str]] = {i: [q.question] for i, q in enumerate(questions)}
        force_answer: set = set()
        active = list(range(len(questions)))
        for step in range(1, self.spec.max_steps + 1):
            if not active:
                break
            last = step == self.spec.max_steps
            prompts_ = []
            for i in active:
                passages = ctx[i][: self.spec.max_passages]
                if last or i in force_answer:
                    prompts_.append(prompts.rag_prompt(questions[i].question, passages, self.max_words))
                else:
                    prompts_.append(
                        prompts.step_prompt(questions[i].question, passages, self.max_words, queries[i])
                    )
            gens = self._generate(prompts_, self.step_tokens)
            searches: List[tuple] = []
            newly_forced: List[int] = []
            for i, g in zip(active, gens):
                r = results[i]
                r.add(g, self.llm.n_params)
                r.trace.append(g.text)
                if last or i in force_answer:
                    r.prediction = prompts.parse_answer(g.text)
                    continue
                kind, payload = prompts.parse_step(g.text)
                if kind == "answer":
                    r.prediction = payload or ""
                elif normalize_answer(payload) in {normalize_answer(q) for q in queries[i]}:
                    force_answer.add(i)
                    newly_forced.append(i)
                else:
                    queries[i].append(payload)
                    searches.append((i, payload))
            if searches:
                new_hits = self._retrieve([q for _, q in searches], [results[i] for i, _ in searches])
                for (i, _), h in zip(searches, new_hits):
                    have = {p.pid for p in ctx[i]}
                    ctx[i].extend(p for p in h if p.pid not in have)
            active = sorted([i for i, _ in searches] + newly_forced)
        for i, r in enumerate(results):
            r.retrieved_pids = [p.pid for p in ctx[i][: self.spec.max_passages]]
        return results
