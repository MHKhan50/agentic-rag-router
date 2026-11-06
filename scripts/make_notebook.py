"""Regenerates notebooks/01_build_oracle_dataset.ipynb from the cells below."""

import json
from pathlib import Path

CELLS = [
    ("markdown", """# Agentic RAG Router — 01: Build the oracle routing dataset

Every benchmark question is answered by all four specialist agents. For each answer we record correctness and the exact cost (LLM calls, tokens, FLOPs, time). The **oracle label** of a question is the cheapest agent that answered it correctly. These labels are what the neural router learns to predict.

| Agent | Generator | Strategy |
|---|---|---|
| `A_direct_small` | Qwen2.5-1.5B-Instruct | no retrieval |
| `B_rag_small` | Qwen2.5-1.5B-Instruct | retrieve once (top-5) |
| `C_rag_large` | Qwen2.5-7B-Instruct | retrieve once (top-5) |
| `D_iterative_large` | Qwen2.5-7B-Instruct | up to 3 reason/retrieve steps |

Retrieval: `intfloat/multilingual-e5-base` (768-d) over one pooled corpus.
Questions: NQ-Open, SQuAD v1.1, HotpotQA (distractor), MuSiQue-Ans (validation splits).

**Kaggle settings:** Accelerator = *GPU T4 x2*, Internet = *On*.
Run with `RUN_MODE = "smoke"` first (~10 min), then switch to `"full"`. The run is resumable: if the session stops, run all cells again and finished questions are skipped."""),
    ("code", """import os, sys, subprocess, json, time, platform
from pathlib import Path

REPO_URL = "https://github.com/MHKhan50/agentic-rag-router"
REPO_DIR = Path("/kaggle/working/agentic-rag-router") if Path("/kaggle").exists() else Path("agentic-rag-router").resolve()
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
else:
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
sys.path.insert(0, str(REPO_DIR / "src"))

import numpy as np, pandas as pd, torch, transformers, datasets
print("python", platform.python_version(), "| torch", torch.__version__,
      "| transformers", transformers.__version__, "| datasets", datasets.__version__)
assert torch.cuda.is_available(), "Turn on a GPU accelerator (Settings > Accelerator > GPU T4 x2)"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"cuda:{i}", p.name, f"{p.total_memory/1e9:.1f} GB")
COMMIT = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print("repo commit", COMMIT)"""),
    ("code", """from agentic_rag_router.config import PipelineConfig

RUN_MODE = "smoke"   # "smoke" -> 10 questions per dataset; "full" -> 600 per dataset

cfg = PipelineConfig.smoke() if RUN_MODE == "smoke" else PipelineConfig()
OUT = Path("/kaggle/working" if Path("/kaggle").exists() else ".") / f"oracle_{RUN_MODE}"
OUT.mkdir(parents=True, exist_ok=True)
print(OUT, cfg.n_per_dataset)"""),
    ("code", """from agentic_rag_router.data import load_questions, build_corpus
from agentic_rag_router.pipeline import write_jsonl

questions = load_questions(cfg.n_per_dataset, cfg.seed)
corpus = build_corpus(questions)
print(len(questions), "questions |", len(corpus), "passages in the pooled corpus")
print(pd.Series([q.dataset for q in questions]).value_counts().to_string())

write_jsonl(OUT / "questions.jsonl", (
    {"qid": q.qid, "dataset": q.dataset, "question": q.question, "answers": q.answers,
     "gold_pids": q.gold_pids, "meta": q.meta} for q in questions))
write_jsonl(OUT / "corpus.jsonl", ({"pid": p.pid, "title": p.title, "text": p.text} for p in corpus))"""),
    ("code", """from agentic_rag_router.embedder import E5Embedder
from agentic_rag_router.retrieval import DenseRetriever

embedder = E5Embedder(cfg.embedding_model, device="cuda:0")

pemb_path = OUT / "passage_emb.npy"
if pemb_path.exists() and np.load(pemb_path, mmap_mode="r").shape[0] == len(corpus):
    passage_emb = np.load(pemb_path)
else:
    t0 = time.time()
    passage_emb = embedder.encode_passages([p.render() for p in corpus])
    np.save(pemb_path, passage_emb)
    print(f"embedded {len(corpus)} passages in {time.time()-t0:.0f}s")
retriever = DenseRetriever(corpus, passage_emb)

# Router input features: one 768-d query embedding per question.
query_emb = embedder.encode_queries([q.question for q in questions])
np.save(OUT / "query_emb.npy", query_emb)
(OUT / "query_emb_qids.json").write_text(json.dumps([q.qid for q in questions]))
print("query embeddings", query_emb.shape)

# Retrieval sanity check: how often the question's own gold passage is in the top 5.
hits = retriever.search(query_emb, 5)
rec = [len(set(q.gold_pids) & {p.pid for p in h}) / len(q.gold_pids) for q, h in zip(questions, hits) if q.gold_pids]
print(f"gold-passage recall@5 (single retrieval): {np.mean(rec):.3f}")"""),
    ("code", """from agentic_rag_router.pipeline import run_agents

t0 = time.time()
runs = run_agents(questions, cfg, retriever, embedder.encode_queries, OUT)
print(f"agents finished in {(time.time()-t0)/60:.1f} min")"""),
    ("code", """from agentic_rag_router.oracle import score_runs, build_oracle, assign_splits, summarise

long = score_runs(questions, runs)
oracle = build_oracle(long, cfg.correct_f1_threshold)
oracle["split"] = assign_splits(oracle, cfg.split_fractions, cfg.seed)
order = oracle.attrs["agent_order"]
print("agent cost order (by measured FLOPs):", order)

print("\\n== Accuracy and cost per agent (all questions) ==")
display(summarise(long, oracle))

print("\\n== F1 per agent per dataset ==")
display(long.pivot_table(index="dataset", columns="agent", values="f1", aggfunc="mean")[order].round(3))

print("\\n== Oracle label distribution per dataset ==")
display(pd.crosstab(oracle["dataset"], oracle["oracle_agent"])[[a for a in order if a in set(oracle["oracle_agent"])]])
print(f"questions no agent answered: {oracle['none_correct'].mean():.1%}")
print(oracle["split"].value_counts().to_string())

long.to_csv(OUT / "agent_results_long.csv", index=False)
oracle.to_csv(OUT / "oracle_labels.csv", index=False)"""),
    ("code", """# Spot-check: predictions next to gold answers for a few questions per dataset.
gold = {q.qid: q.answers[:3] for q in questions}
sample = long[long["qid"].isin(oracle.groupby("dataset").head(2)["qid"])]
view = sample.pivot(index="qid", columns="agent", values="prediction")[order]
view.insert(0, "gold", view.index.map(gold))
pd.set_option("display.max_colwidth", 60)
display(view)"""),
    ("code", """# The router consumes exactly these features and labels. Quick shape/forward check.
from agentic_rag_router.router import DynamicNeuralRouter

qid_index = {qid: i for i, qid in enumerate(json.loads((OUT / "query_emb_qids.json").read_text()))}
X = torch.tensor(query_emb[[qid_index[q] for q in oracle["qid"]]])
y = torch.tensor(oracle["oracle_label"].values)
router = DynamicNeuralRouter(d_model=X.shape[1], num_agents=len(order))
logits = router(X)
print("features", tuple(X.shape), "| labels", tuple(y.shape), "| logits", tuple(logits.shape))
print("untrained loss", torch.nn.functional.cross_entropy(logits, y).item(), "(~ln 4 = 1.386 expected)")"""),
    ("code", """import shutil
manifest = {
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "repo_commit": COMMIT,
    "run_mode": RUN_MODE,
    "config": cfg.to_dict(),
    "agent_cost_order": order,
    "n_questions": len(questions),
    "n_passages": len(corpus),
    "versions": {"python": platform.python_version(), "torch": torch.__version__,
                 "transformers": transformers.__version__, "datasets": datasets.__version__},
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
archive = shutil.make_archive(str(OUT), "zip", OUT)
print("saved", archive)
print(sorted(p.name for p in OUT.iterdir()))"""),
    ("markdown", """### Outputs (in `oracle_<mode>/`)
- `questions.jsonl`, `corpus.jsonl` — sampled questions and the pooled passage corpus
- `runs/<agent>.jsonl` — raw agent outputs with full cost accounting and step traces
- `agent_results_long.csv` — one row per (question, agent): EM, F1, Acc, cost
- `oracle_labels.csv` — one row per question: oracle agent, per-agent correctness/FLOPs, split
- `query_emb.npy` + `query_emb_qids.json` — 768-d router features
- `manifest.json` — config, versions, hardware, repo commit

**Next:** train the router and the baselines on `oracle_labels.csv` + `query_emb.npy` (notebook 02)."""),
]


def main() -> None:
    nb = {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": src.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, src in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parents[1] / "notebooks" / "01_build_oracle_dataset.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
