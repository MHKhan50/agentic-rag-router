"""End-to-end oracle dataset construction.

Runs every agent over every question and appends results to
``<out>/runs/<agent>.jsonl`` one batch at a time, so a Kaggle session that
stops halfway can simply be re-run: finished questions are skipped.
Agents that share a generator model run back-to-back so each model is
loaded once.

The full run needs more than one 12-hour Kaggle session. On Kaggle the
pipeline therefore (1) picks up ``runs/*.jsonl`` from an earlier version of
the notebook attached as an input, and (2) stops starting new batches after
a time budget (11 h by default), saving everything finished so far.
"""

from __future__ import annotations

import json
import os
import time
from itertools import groupby
from pathlib import Path
from typing import Callable, Dict, List

from .agents import Agent
from .config import SMALL_LLM, PipelineConfig
from .data import Question
from .retrieval import DenseRetriever

SESSION_START = time.time()  # the package is imported in the first minute of a Kaggle session
ON_KAGGLE = Path("/kaggle/input").exists()


class TimeBudgetReached(RuntimeError):
    """Raised after saving partial results, so the notebook stops before the scoring cells."""


def default_time_budget_s() -> float | None:
    hours = os.environ.get("ARR_TIME_BUDGET_HOURS")
    if hours:
        return float(hours) * 3600
    return 11 * 3600 if ON_KAGGLE else None


def import_previous_runs(out_dir: Path, questions: List[Question], agent_names: List[str],
                         search_root: Path | None = None, log: Callable[[str], None] = print) -> None:
    """Merge rows from ``runs/<agent>.jsonl`` files under ``search_root`` (an earlier
    version's output attached as a Kaggle input) into ``out_dir/runs``. Only rows
    for the current questions are kept, and each question is kept once."""
    root = search_root if search_root is not None else (Path("/kaggle/input") if ON_KAGGLE else None)
    if root is None or not root.exists():
        return
    want = {q.qid for q in questions}
    for name in agent_names:
        dest = out_dir / "runs" / f"{name}.jsonl"
        have = {r["qid"] for r in read_jsonl(dest)}
        added = []
        for cand in sorted(root.glob(f"**/runs/{name}.jsonl")):
            if cand.resolve() == dest.resolve():
                continue
            for r in read_jsonl(cand):
                if r["qid"] in want and r["qid"] not in have:
                    added.append(r)
                    have.add(r["qid"])
        if added:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("a", encoding="utf-8") as f:
                for r in added:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log(f"[resume] {name}: {len(added)} answers from an earlier run, {len(have)}/{len(want)} now done")


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
    time_budget_s: float | None = None,
    previous_runs_root: Path | None = None,
) -> Dict[str, List[dict]]:
    """Run (or resume) all agents; returns {agent_name: [result dicts]}.

    Raises ``TimeBudgetReached`` (after saving) if the budget runs out first."""
    if time_budget_s is None:
        time_budget_s = default_time_budget_s()
    import_previous_runs(out_dir, questions, [a.name for a in cfg.agents], previous_runs_root, log)
    stopped = False
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
        elif stopped:
            pass
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
                    if time_budget_s is not None and time.time() - SESSION_START > time_budget_s:
                        stopped = True
                        break
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
                if stopped:
                    break
            if hasattr(llm, "close"):
                llm.close()
        for spec in group:
            rows = read_jsonl(out_dir / "runs" / f"{spec.name}.jsonl")
            keep = {q.qid for q in questions}
            runs[spec.name] = list({r["qid"]: r for r in rows if r["qid"] in keep}.values())
    if stopped:
        left = {name: len(questions) - len(rows) for name, rows in runs.items() if len(rows) < len(questions)}
        missing = [a.name for a in cfg.agents if a.name not in runs]
        raise TimeBudgetReached(
            f"Time budget reached; finished answers are saved in {out_dir / 'runs'}. "
            f"Still to do: {left}{' plus ' + str(missing) if missing else ''}. "
            "Attach this version's output as an input (Add Input -> Your Work) and run the notebook again; "
            "it continues from here.")
    return runs
