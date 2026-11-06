"""End-to-end oracle dataset construction.

Runs every agent over every question and appends results to
``<out>/runs/<agent>.jsonl`` one batch at a time, so a Kaggle session that
stops halfway can simply be re-run: finished questions are skipped.
Agents that share a generator model run back-to-back so each model is
loaded once.
"""

from __future__ import annotations

import json
import time
from itertools import groupby
from pathlib import Path
from typing import Callable, Dict, List

from .agents import Agent
from .config import SMALL_LLM, PipelineConfig
from .data import Question
from .retrieval import DenseRetriever


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_agents(
    questions: List[Question],
    cfg: PipelineConfig,
    retriever: DenseRetriever,
    encode_queries: Callable,
    out_dir: Path,
    llm_factory: Callable | None = None,
    chunk_size: int = 64,
    log: Callable[[str], None] = print,
) -> Dict[str, List[dict]]:
    """Run (or resume) all agents; returns {agent_name: [result dicts]}."""
    if llm_factory is None:
        from .llm import HFChatLLM

        def llm_factory(model_id: str):
            return HFChatLLM(model_id, device_map={"": 0} if model_id == SMALL_LLM else "auto")

    runs: Dict[str, List[dict]] = {}
    specs = sorted(cfg.agents, key=lambda s: s.model_id != SMALL_LLM)  # small model first
    for model_id, group in groupby(specs, key=lambda s: s.model_id):
        group = list(group)
        pending = {}
        for spec in group:
            path = out_dir / "runs" / f"{spec.name}.jsonl"
            done = {r["qid"] for r in read_jsonl(path)}
            pending[spec.name] = [q for q in questions if q.qid not in done]
        if not any(pending.values()):
            log(f"[skip] {model_id}: all agents already complete")
        else:
            log(f"[load] {model_id}")
            llm = llm_factory(model_id)
            for spec in group:
                todo = pending[spec.name]
                path = out_dir / "runs" / f"{spec.name}.jsonl"
                agent = Agent(
                    spec, llm, retriever, encode_queries,
                    passage_max_words=cfg.passage_max_words,
                    answer_max_new_tokens=cfg.answer_max_new_tokens,
                    step_max_new_tokens=cfg.step_max_new_tokens,
                )
                t0, n_done = time.perf_counter(), 0
                for start in range(0, len(todo), chunk_size):
                    chunk = todo[start : start + chunk_size]
                    results = agent.run(chunk)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("a", encoding="utf-8") as f:
                        for r in results:
                            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
                    n_done += len(chunk)
                    rate = (time.perf_counter() - t0) / n_done
                    eta = rate * (len(todo) - n_done) / 60
                    log(f"[{spec.name}] {n_done}/{len(todo)}  {rate:.2f}s/q  eta {eta:.1f} min")
            if hasattr(llm, "close"):
                llm.close()
        for spec in group:
            rows = read_jsonl(out_dir / "runs" / f"{spec.name}.jsonl")
            keep = {q.qid for q in questions}
            runs[spec.name] = list({r["qid"]: r for r in rows if r["qid"] in keep}.values())
    return runs
