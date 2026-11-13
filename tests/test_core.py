"""Unit tests for everything that does not need a GPU or model download."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rag_router import prompts  # noqa: E402
from agentic_rag_router.agents import Agent  # noqa: E402
from agentic_rag_router.config import AgentSpec, PipelineConfig  # noqa: E402
from agentic_rag_router.data import (  # noqa: E402
    build_corpus,
    parse_hotpotqa,
    parse_musique,
    parse_nq_open,
    parse_squad,
    sample_questions,
)
from agentic_rag_router.llm import Generation  # noqa: E402
from agentic_rag_router.metrics import score  # noqa: E402
from agentic_rag_router.oracle import assign_splits, build_oracle, score_runs, summarise  # noqa: E402
from agentic_rag_router.pipeline import TimeBudgetReached, read_jsonl, run_agents  # noqa: E402
from agentic_rag_router.retrieval import DenseRetriever  # noqa: E402

# Rows mirror the published Hugging Face schemas of each dataset.
NQ_ROW = {"question": "who wrote hamlet", "answer": ["William Shakespeare", "Shakespeare"]}
SQUAD_ROW = {
    "id": "5733be284776f41900661182",
    "title": "University_of_Notre_Dame",
    "context": "Notre Dame is in  Indiana.",
    "question": "Where is Notre Dame?",
    "answers": {"text": ["Indiana", "Indiana"], "answer_start": [17, 17]},
}
HOTPOT_ROW = {
    "id": "5a8b57f25542995d1e6f1371",
    "question": "Which city is the birthplace of the author of Hamlet?",
    "answer": "Stratford-upon-Avon",
    "type": "bridge",
    "level": "hard",
    "supporting_facts": {"title": ["Hamlet", "William Shakespeare"], "sent_id": [0, 0]},
    "context": {
        "title": ["Hamlet", "William Shakespeare", "Macbeth"],
        "sentences": [
            ["Hamlet is a tragedy by William Shakespeare."],
            ["Shakespeare was born in ", "Stratford-upon-Avon."],
            ["Macbeth is a tragedy."],
        ],
    },
}
MUSIQUE_ROW = {
    "id": "2hop__482757_12019",
    "question": "When was the person born?",
    "answer": "1994",
    "answer_aliases": ["1994 AD"],
    "answerable": True,
    "paragraphs": [
        {"idx": 0, "title": "A", "paragraph_text": "A was born in 1994.", "is_supporting": True},
        {"idx": 1, "title": "B", "paragraph_text": "B is a city.", "is_supporting": False},
    ],
}


# ---------------------------------------------------------------- metrics
def test_metrics_standard_cases():
    s = score("The Shakespeare.", ["William Shakespeare", "Shakespeare"])
    assert s["em"] == 1.0 and s["f1"] == 1.0 and s["acc"] == 1.0
    s = score("William", ["William Shakespeare"])
    assert s["em"] == 0.0 and s["f1"] == pytest.approx(2 / 3) and s["acc"] == 0.0
    assert score("", ["x"])["f1"] == 0.0
    assert score("3", ["three"])["em"] == 1.0  # number words normalised


def test_correctness_rule_rejects_half_overlap():
    from agentic_rag_router.metrics import is_correct

    assert not is_correct(score("Articles 101(3) and 102", ["Articles 106 and 107"]))  # F1 = 0.5
    assert is_correct(score("Thorgan Hazard", ["Thorgan Ganael Francis Hazard"]))  # F1 = 0.67
    assert is_correct(score("to evade being drafted into the army", ["being drafted"]))  # contained


# ---------------------------------------------------------------- data
def test_parsers_follow_schemas():
    nq = parse_nq_open(NQ_ROW, 7)
    assert nq.qid == "nq_open-7" and nq.passages == [] and len(nq.answers) == 2

    sq = parse_squad(SQUAD_ROW, 0)
    assert sq.answers == ["Indiana"]  # duplicates removed
    assert sq.passages[0].title == "University of Notre Dame"
    assert sq.passages[0].text == "Notre Dame is in Indiana."  # whitespace normalised
    assert sq.gold_pids == [sq.passages[0].pid]

    hp = parse_hotpotqa(HOTPOT_ROW, 0)
    assert len(hp.passages) == 3 and len(hp.gold_pids) == 2
    assert hp.passages[1].text == "Shakespeare was born in Stratford-upon-Avon."

    mu = parse_musique(MUSIQUE_ROW, 0)
    assert mu.answers == ["1994", "1994 AD"] and len(mu.gold_pids) == 1
    assert mu.meta["hops"] == "2hop"
    assert parse_musique({**MUSIQUE_ROW, "answerable": False}, 0) is None


def test_sampling_is_seeded_and_corpus_deduplicates():
    rows = [{**SQUAD_ROW, "id": f"id{i}"} for i in range(20)]
    a = sample_questions(rows, "squad", 5, seed=1)
    b = sample_questions(rows, "squad", 5, seed=1)
    c = sample_questions(rows, "squad", 5, seed=2)
    assert [q.qid for q in a] == [q.qid for q in b]
    assert [q.qid for q in a] != [q.qid for q in c]
    assert len(build_corpus(a)) == 1  # same context shared by all five


# ---------------------------------------------------------------- retrieval
def test_dense_retriever_ranks_by_cosine():
    hp = parse_hotpotqa(HOTPOT_ROW, 0)
    emb = np.eye(3, dtype=np.float32)
    r = DenseRetriever(hp.passages, emb)
    hits = r.search(np.array([[0.1, 0.9, 0.3]]), top_k=2)[0]
    assert [p.title for p in hits] == ["William Shakespeare", "Macbeth"]


# ---------------------------------------------------------------- prompts
def test_answer_and_step_parsing():
    assert prompts.parse_answer("Answer: Stratford-upon-Avon.") == "Stratford-upon-Avon"
    assert prompts.parse_answer("Paris") == "Paris"
    assert prompts.parse_step("Search: birthplace of Shakespeare") == ("search", "birthplace of Shakespeare")
    assert prompts.parse_step("Answer: 1994") == ("answer", "1994")
    # Failure modes seen in real runs:
    assert prompts.parse_answer("Answer: Answer: Leia Organa") == "Leia Organa"
    assert prompts.parse_step("Answer: Search: release companies of Cinderella") == (
        "search",
        "release companies of Cinderella",
    )


# ---------------------------------------------------------------- agents
class FakeLLM:
    """Scripted generator: returns the next reply for each prompt."""

    model_id = "fake"
    n_params = 1_000

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def generate(self, prompts_, max_new_tokens, batch_size):
        self.calls += len(prompts_)
        return [Generation(self.reply(m), prompt_tokens=10, gen_tokens=2, seconds=0.01) for m in prompts_]


def _fake_world():
    hp = parse_hotpotqa(HOTPOT_ROW, 0)
    passages = hp.passages
    titles = [p.title.lower() for p in passages]

    def encode(texts):
        out = np.zeros((len(texts), len(passages)), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, title in enumerate(titles):
                out[i, j] = 1.0 if title.split()[-1] in t.lower() else 0.01 * (j + 1)
        return out

    return hp, DenseRetriever(passages, np.eye(len(passages), dtype=np.float32)), encode


def test_iterative_agent_searches_then_answers():
    q, retriever, encode = _fake_world()

    def reply(messages):
        text = messages[-1]["content"]
        if "Stratford" in text:
            return "Answer: Stratford-upon-Avon"
        return "Search: where was Shakespeare born"

    spec = AgentSpec("D", "fake", "iterative_rag", top_k=1, max_steps=3, max_passages=5)
    llm = FakeLLM(reply)
    [res] = Agent(spec, llm, retriever, encode).run([q])
    assert res.prediction == "Stratford-upon-Avon"
    assert res.llm_calls == 2 and res.retrieval_calls == 2
    assert res.flops == pytest.approx(2 * 1_000 * 12 * 2)


def test_iterative_agent_forces_answer_on_last_step():
    q, retriever, encode = _fake_world()
    spec = AgentSpec("D", "fake", "iterative_rag", top_k=1, max_steps=2, max_passages=5)
    llm = FakeLLM(lambda m: "Search: something else")
    [res] = Agent(spec, llm, retriever, encode).run([q])
    assert res.llm_calls == 2  # step 1 searched, step 2 had to answer
    assert res.retrieval_calls == 2


def test_iterative_agent_stops_repeated_searches():
    q, retriever, encode = _fake_world()

    spec = AgentSpec("E", "fake", "iterative_rag", top_k=1, max_steps=3, max_passages=5)
    llm = FakeLLM(lambda m: "Search: " + q.question + "?" if "Searches already made" in m[-1]["content"] else "Answer: Stratford")
    [res] = Agent(spec, llm, retriever, encode).run([q])
    # Step 1 repeats the original question -> no new retrieval; step 2 must answer.
    assert res.retrieval_calls == 1 and res.llm_calls == 2
    assert res.prediction == "Stratford"


def test_direct_agent_needs_no_retriever():
    q, _, _ = _fake_world()
    [res] = Agent(AgentSpec("A", "fake", "direct"), FakeLLM(lambda m: "Answer: x")).run([q])
    assert res.prediction == "x" and res.retrieval_calls == 0


def test_rag_agent_requires_retriever():
    with pytest.raises(ValueError):
        Agent(AgentSpec("B", "fake", "single_rag", top_k=2), FakeLLM(lambda m: ""))


# ---------------------------------------------------------------- oracle + pipeline
def test_pipeline_oracle_and_resume(tmp_path):
    questions = [parse_hotpotqa({**HOTPOT_ROW, "id": f"h{i}"}, i) for i in range(10)]
    _, retriever, encode = _fake_world()
    cfg = PipelineConfig(
        agents=[
            AgentSpec("cheap", "small", "direct"),
            AgentSpec("rich", "big", "single_rag", top_k=2, max_passages=2),
        ]
    )
    # The cheap agent only knows even-numbered questions; the rich one knows all.
    def factory(model_id):
        if model_id == "small":
            return FakeLLM(lambda m: "Answer: Stratford-upon-Avon")
        llm = FakeLLM(lambda m: "Answer: Stratford-upon-Avon")
        llm.n_params = 50_000
        return llm

    runs = run_agents(questions, cfg, retriever, encode, tmp_path, llm_factory=factory, chunk_size=4, log=lambda s: None)
    # Flip half the cheap answers to wrong, then label.
    for r in runs["cheap"]:
        if int(r["qid"].split("h")[-1]) % 2:
            r["prediction"] = "London"
    long = score_runs(questions, runs)
    oracle = build_oracle(long)
    assert oracle.attrs["agent_order"] == ["cheap", "rich"]
    counts = oracle["oracle_agent"].value_counts().to_dict()
    assert counts == {"cheap": 5, "rich": 5}
    assert not oracle["none_correct"].any()

    splits = assign_splits(oracle, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    assert splits.value_counts().to_dict() == {"train": 7, "val": 2, "test": 1}

    table = summarise(long, oracle)
    assert table.loc["ORACLE_ROUTER", "f1"] == 1.0
    assert table.loc["ORACLE_ROUTER", "gflops"] < table.loc["rich", "gflops"]

    # Re-running loads nothing: every question is already on disk.
    def boom(model_id):
        raise AssertionError("model should not be reloaded")

    again = run_agents(questions, cfg, retriever, encode, tmp_path, llm_factory=boom, log=lambda s: None)
    assert {k: len(v) for k, v in again.items()} == {"cheap": 10, "rich": 10}



def test_pipeline_resumes_from_earlier_session_and_stops_on_budget(tmp_path):
    questions = [parse_hotpotqa({**HOTPOT_ROW, "id": f"h{i}"}, i) for i in range(10)]
    _, retriever, encode = _fake_world()
    cfg = PipelineConfig(agents=[AgentSpec("cheap", "small", "direct"), AgentSpec("rich", "big", "direct")])
    calls = []

    def factory(model_id):
        calls.append(model_id)
        return FakeLLM(lambda m: "Answer: Stratford-upon-Avon")

    # Session 1 finished the first agent and 6 of 10 questions for the second.
    first = tmp_path / "session1"
    run_agents(questions, cfg, retriever, encode, first, llm_factory=factory, chunk_size=4, log=lambda s: None)
    rich = read_jsonl(first / "runs" / "rich.jsonl")
    (first / "runs" / "rich.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rich[:6]))

    # Session 2: a fresh working directory, with session 1's output attached as an input.
    second = tmp_path / "session2"
    calls.clear()
    runs = run_agents(questions, cfg, retriever, encode, second, llm_factory=factory, chunk_size=4,
                      log=lambda s: None, previous_runs_root=tmp_path / "session1")
    assert calls == ["big"]  # the first agent was complete, so only the second model loaded
    assert {k: len(v) for k, v in runs.items()} == {"cheap": 10, "rich": 10}

    # A zero budget stops before any batch and says what is left.
    third = tmp_path / "session3"
    with pytest.raises(TimeBudgetReached, match="Still to do"):
        run_agents(questions, cfg, retriever, encode, third, llm_factory=factory, chunk_size=4,
                   log=lambda s: None, time_budget_s=0)


# ---------------------------------------------------------------- evaluation + baselines
from agentic_rag_router import baselines, evaluation as ev  # noqa: E402


def test_policy_metrics_and_policies():
    correct = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]])
    flops = np.tile([1e9, 2e9, 4e9], (4, 1)).astype(float)
    oracle = np.array([0, 1, 2, 2])
    m = ev.policy_metrics(oracle, correct, flops, oracle)
    assert m["accuracy"] == 0.75 and m["routing_accuracy"] == 1.0
    assert m["gflops"] == pytest.approx((1 + 2 + 4 + 4) / 4)
    assert m["cost_vs_strongest"] == pytest.approx(11 / 16)

    probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]])
    assert ev.confidence_escalation(probs, 0.5).tolist() == [0, 2]
    pc = np.array([[0.2, 0.8, 0.9], [0.1, 0.2, 0.3]])
    assert ev.cheapest_confident(pc, 0.7).tolist() == [1, 2]


def test_pareto_auc_bootstrap_mcnemar_ece():
    front = ev.pareto_front([(1, 0.5), (2, 0.4), (3, 0.7), (3, 0.6)])
    assert front == [(1, 0.5), (3, 0.7)]
    assert ev.area_under_curve([(1, 0.5), (3, 0.7)], max_cost=4) == pytest.approx((0 * 1 + 0.5 * 2 + 0.7 * 1) / 4)
    lo, hi = ev.bootstrap_ci(np.array([0, 1] * 50, dtype=float))
    assert lo < 0.5 < hi
    a, b = np.array([1] * 10 + [0] * 10), np.array([0] * 10 + [0] * 10)
    r = ev.mcnemar_exact(a, b)
    assert r["a_only"] == 10 and r["p_value"] == pytest.approx(2 / 2**10)
    assert ev.mcnemar_exact(a, a)["p_value"] == 1.0
    assert ev.expected_calibration_error(np.array([0.9, 0.9]), np.array([1, 1])) == pytest.approx(0.1)


def test_baselines_shapes_and_knn_vote():
    X = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
    y = np.array([0, 0, 2, 2])
    p = baselines.knn_proba(X, y, np.array([[1, 0.05]]), k_agents=3, k=2)
    assert p.tolist() == [[1.0, 0.0, 0.0]]
    lr = baselines.fit_logistic_regression(X, y, k_agents=3)(X)
    assert lr.shape == (4, 3) and np.allclose(lr.sum(1), 1) and (lr[:, 1] == 0).all()
    assert baselines.majority_class(np.array([1, 1, 2]), 3, 3).tolist() == [1, 1, 1]
