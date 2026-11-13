"""Regenerates the Kaggle notebooks in notebooks/ from the cell lists below."""

import json
from pathlib import Path

NB01 = [
    ("markdown", """# Agentic RAG Router — 01: Build the oracle routing dataset

Every benchmark question is answered by all five specialist agents. For each answer we record correctness and the exact cost (LLM calls, tokens, FLOPs, time). The **oracle label** of a question is the cheapest agent that answered it correctly. These labels are what the neural router learns to predict.

| Agent | Generator | Strategy |
|---|---|---|
| `A_direct_small` | Qwen2.5-1.5B-Instruct | no retrieval |
| `B_direct_large` | Qwen2.5-7B-Instruct | no retrieval |
| `C_rag_small` | Qwen2.5-1.5B-Instruct | retrieve once (top-5) |
| `D_rag_large` | Qwen2.5-7B-Instruct | retrieve once (top-5) |
| `E_iterative_large` | Qwen2.5-7B-Instruct | up to 3 reason/retrieve steps |

A prediction is **correct** if it matches exactly, contains the gold answer, or has token F1 ≥ 0.6.

Retrieval: `intfloat/multilingual-e5-base` (768-d) over one pooled corpus.
Questions: NQ-Open, SQuAD v1.1, HotpotQA (distractor), MuSiQue-Ans (validation splits).

**Kaggle settings:** Accelerator = *GPU T4 x2*, Internet = *On*.
Run with `RUN_MODE = "smoke"` first (~15 min), then switch to `"full"` (2,000 questions) and start it with **Save Version → Save & Run All (Commit)** so it runs in the background.

The full run needs about 13–18 GPU hours on T4 x2, more than one 12-hour Kaggle session. The pipeline stops starting new batches after 11 hours and saves everything finished (the version then shows as failed with a `TimeBudgetReached` message, which is expected). To continue: **Add Input → Your Work → this notebook** (its latest output), then **Save & Run All** again. Finished answers are picked up from the attached output and only the rest is run."""),
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

RUN_MODE = "smoke"   # "smoke" -> 10 questions per dataset; "full" -> 500 per dataset

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
display(summarise(long, oracle, cfg.correct_f1_threshold))

from agentic_rag_router.oracle import add_correct
print("\\n== Accuracy (correct) per agent per dataset ==")
display(add_correct(long, cfg.correct_f1_threshold).pivot_table(index="dataset", columns="agent", values="correct", aggfunc="mean")[order].round(3))

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
print("untrained loss", torch.nn.functional.cross_entropy(logits, y).item(), "(~ln K expected)", f"K={len(order)}")"""),
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


NB02 = [
    ("markdown", """# Agentic RAG Router — 02: Train the router and compare against baselines

Input: the oracle dataset from notebook 01 (`oracle_labels.csv`, `query_emb.npy`). Either run this in the same session after notebook 01, or attach notebook 01's output with **Add Input → Notebook output**.

**What is compared (test split only, never used for training or model selection):**
- fixed policies: always one agent (each of the five), random, majority label, and the oracle (upper bound)
- learned baselines on the same 768-d embeddings: cosine kNN, logistic regression
- the neural router, two objectives: *oracle* (softmax over agents) and *correctness* (P(correct) per agent), plus a no-gate ablation
- 5 seeds per neural model; 95% bootstrap CIs; exact McNemar tests; calibration (ECE)

**Operating point.** Each learned policy has a threshold τ that trades accuracy for compute. τ* is chosen **on the validation split** as the cheapest setting that stays within 1 point of the strongest agent's validation accuracy, then applied unchanged to the test split.

Accelerator: GPU or CPU both work (the router is small)."""),
    NB01[1],
    ("code", """import glob
work = Path("/kaggle/working" if Path("/kaggle").exists() else ".")
cands = sorted(glob.glob("/kaggle/input/**/oracle_labels.csv", recursive=True)) + sorted(str(p) for p in work.glob("oracle_*/oracle_labels.csv"))
assert cands, "No oracle_labels.csv found: run notebook 01 first, or add its output as an input."
DATA_DIR = Path(([c for c in cands if "oracle_full" in c] or cands)[0]).parent
print("using", DATA_DIR)

oracle = pd.read_csv(DATA_DIR / "oracle_labels.csv")
data_manifest = json.loads((DATA_DIR / "manifest.json").read_text())
ORDER = data_manifest["agent_cost_order"]
qids = json.loads((DATA_DIR / "query_emb_qids.json").read_text())
emb = np.load(DATA_DIR / "query_emb.npy")
pos = {q: i for i, q in enumerate(qids)}

X = emb[[pos[q] for q in oracle["qid"]]].astype(np.float32)
C = oracle[[f"correct__{a}" for a in ORDER]].to_numpy(dtype=int)      # n x K, 1 = agent answered correctly
FL = oracle[[f"flops__{a}" for a in ORDER]].to_numpy(dtype=float)     # n x K, measured compute
y = oracle["oracle_label"].to_numpy(dtype=int)                         # cheapest correct agent
K = len(ORDER)
S = {s: np.where(oracle["split"].to_numpy() == s)[0] for s in ["train", "val", "test"]}
tr, va, te = S["train"], S["val"], S["test"]
print({s: len(i) for s, i in S.items()}, "| agents (cheap -> strong):", ORDER)
if len(tr) < 500:
    print("WARNING: smoke-sized data. This run only checks the code path; the numbers are not results.")
SEEDS = [0, 1, 2, 3, 4]"""),
    ("code", """from agentic_rag_router import baselines as bl, evaluation as ev

choices = {}   # policy name -> list of test choice arrays (one per seed)
probs = {}     # learned policy name -> list of (val_scores, test_scores), one per seed

for k, a in enumerate(ORDER):
    choices[f"always {a}"] = [bl.fixed(len(te), k)]
choices["random"] = [bl.random_uniform(len(te), K, s) for s in SEEDS]
choices["majority label"] = [bl.majority_class(y[tr], len(te), K)]
choices["oracle (upper bound)"] = [y[te]]

probs["kNN (k=15)"] = [(bl.knn_proba(X[tr], y[tr], X[va], K), bl.knn_proba(X[tr], y[tr], X[te], K))]
lr = bl.fit_logistic_regression(X[tr], y[tr], K)
probs["logistic regression"] = [(lr(X[va]), lr(X[te]))]
print("baselines ready")"""),
    ("code", """from agentic_rag_router.train import TrainConfig, train_router, predict

VARIANTS = {
    "router: oracle": TrainConfig(objective="oracle"),
    "router: correctness": TrainConfig(objective="correctness"),
    "router: oracle, no gate (ablation)": TrainConfig(objective="oracle", use_gate=False),
}
models = {}
for name, tc in VARIANTS.items():
    T = y if tc.objective == "oracle" else C
    probs[name], models[name] = [], []
    for s in SEEDS:
        m, hist = train_router(X[tr], T[tr], X[va], T[va], K, tc, seed=s)
        probs[name].append((predict(m, X[va], tc.objective), predict(m, X[te], tc.objective)))
        models[name].append((m, hist))
    best_ep = [min(h, key=lambda r: r["val_loss"])["epoch"] for _, h in models[name]]
    print(f"{name}: best epochs per seed {best_ep}")"""),
    ("code", """TAUS = np.linspace(0, 1, 41)

def policy_fn(name):
    return ev.cheapest_confident if "correctness" in name else ev.confidence_escalation

def pick_tau(name, val_scores):
    \"\"\"Cheapest tau (on validation) within 1 point of the strongest agent's validation accuracy.\"\"\"
    fn, target = policy_fn(name), C[va][:, -1].mean() - 0.01
    stats = [(float(t), ev.policy_metrics(fn(val_scores, float(t)), C[va], FL[va])) for t in TAUS]
    ok = [(t, m["gflops"]) for t, m in stats if m["accuracy"] >= target]
    return min(ok, key=lambda x: x[1])[0] if ok else max(stats, key=lambda x: x[1]["accuracy"])[0]

tau_star = {}
for name, runs in probs.items():
    fn = policy_fn(name)
    default_tau = 0.5 if "correctness" in name else 0.0   # 0.0 = plain argmax
    choices[f"{name} (default)"] = [fn(pt, default_tau) for _, pt in runs]
    taus = [pick_tau(name, pv) for pv, _ in runs]
    tau_star[name] = taus
    choices[f"{name} @ tau*"] = [fn(pt, t) for (_, pt), t in zip(runs, taus)]

rows = []
for name, cs in choices.items():
    ms = [ev.policy_metrics(c, C[te], FL[te], y[te]) for c in cs]
    acc = np.array([m["accuracy"] for m in ms])
    lo, hi = ev.bootstrap_ci(C[te][np.arange(len(te)), cs[0]].astype(float))
    rows.append({
        "policy": name, "seeds": len(ms),
        "accuracy": acc.mean(), "acc_std": acc.std(), "acc_ci95": f"[{lo:.3f}, {hi:.3f}]",
        "gflops/query": np.mean([m["gflops"] for m in ms]),
        "cost_vs_strongest": np.mean([m["cost_vs_strongest"] for m in ms]),
        "routing_acc": np.mean([m["routing_accuracy"] for m in ms]),
    })
results = pd.DataFrame(rows).set_index("policy").round(4)
print("Test split. CI = 95% bootstrap over questions (seed 0). cost_vs_strongest = compute relative to always using the strongest agent.")
display(results)
print("tau* chosen on validation:", {k: [round(t, 3) for t in v] for k, v in tau_star.items()})"""),
    ("code", """import matplotlib.pyplot as plt

max_cost = FL[te][:, -1].mean() / 1e9
curves, auc = {}, {}
for name, runs in probs.items():
    per_seed = []
    for _, pt in runs:
        pts = ev.sweep(policy_fn(name), pt, C[te], FL[te], TAUS)
        per_seed.append(ev.area_under_curve([(c, a) for _, c, a in pts], max_cost))
    curves[name] = ev.pareto_front([(c, a) for _, c, a in ev.sweep(policy_fn(name), runs[0][1], C[te], FL[te], TAUS)])
    auc[name] = (float(np.mean(per_seed)), float(np.std(per_seed)))

fig, ax = plt.subplots(figsize=(7.5, 5))
for name, front in curves.items():
    cs, accs = zip(*front)
    ax.step(cs, accs, where="post", label=f"{name}  (AUC {auc[name][0]:.3f})")
for k, a in enumerate(ORDER):
    m = ev.policy_metrics(bl.fixed(len(te), k), C[te], FL[te])
    ax.scatter(m["gflops"], m["accuracy"], marker="s", s=40, zorder=3)
    ax.annotate(a, (m["gflops"], m["accuracy"]), textcoords="offset points", xytext=(5, -10), fontsize=8)
m = ev.policy_metrics(y[te], C[te], FL[te])
ax.scatter(m["gflops"], m["accuracy"], marker="*", s=160, color="k", zorder=4, label="oracle router")
ax.set_xlabel("mean GFLOPs per query"); ax.set_ylabel("answer accuracy (correct)")
ax.set_title("Accuracy vs compute on the test split"); ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(work / "pareto_test.png", dpi=200); plt.show()
print("normalised area under the accuracy-cost front (mean ± std over seeds):")
print({k: f"{m:.4f} ± {s:.4f}" for k, (m, s) in auc.items()})"""),
    ("code", """def hits(policy, seed=0):
    return C[te][np.arange(len(te)), choices[policy][seed]]

strongest = f"always {ORDER[-1]}"
tests = {}
for name in ["router: oracle @ tau*", "router: correctness @ tau*"]:
    for ref in [strongest, "logistic regression @ tau*", "kNN (k=15) @ tau*"]:
        tests[f"{name} vs {ref}"] = ev.mcnemar_exact(hits(name), hits(ref))
display(pd.DataFrame(tests).T)

ece = [ev.expected_calibration_error(pt, C[te]) for _, pt in probs["router: correctness"]]
print(f"ECE of P(correct) from the correctness router (all agents, test): {np.mean(ece):.4f} ± {np.std(ece):.4f}")"""),
    ("code", """import torch
# Export the single best router (by validation AUC) for release.
best = None
for name in ["router: oracle", "router: correctness"]:
    for s, ((pv, _), (model, _)) in enumerate(zip(probs[name], models[name])):
        pts = ev.sweep(policy_fn(name), pv, C[va], FL[va], TAUS)
        a = ev.area_under_curve([(c, acc) for _, c, acc in pts], FL[va][:, -1].mean() / 1e9)
        if best is None or a > best[0]:
            best = (a, name, s, model)
val_auc, name, seed, model = best
tc = VARIANTS[name]
EXPORT = work / "router_release"
EXPORT.mkdir(exist_ok=True)
torch.save(model.state_dict(), EXPORT / "router.pt")
(EXPORT / "router_config.json").write_text(json.dumps({
    "agents": ORDER,
    "embedding_model": data_manifest["config"]["embedding_model"],
    "d_model": int(X.shape[1]), "num_agents": K,
    "train_config": tc.to_dict(), "seed": seed,
    "policy": "cheapest_confident" if tc.objective == "correctness" else "confidence_escalation",
    "tau": tau_star[name][seed], "val_auc": val_auc,
    "data_commit": data_manifest["repo_commit"], "code_commit": COMMIT,
}, indent=2))
results.to_csv(work / "results_test.csv")
(work / "results.json").write_text(json.dumps({
    "auc": auc, "tau_star": tau_star, "mcnemar": tests, "ece_correctness": ece,
    "splits": {k: int(len(v)) for k, v in S.items()}, "data_dir": str(DATA_DIR),
}, indent=2, default=float))
print(f"exported {name} (seed {seed}, val AUC {val_auc:.4f}) ->", EXPORT)"""),
    ("markdown", """### Reading the results
- **accuracy / gflops** of `router … @ tau*` against `always <strongest>` is the headline: same accuracy (within 1 point, by construction on validation) at what fraction of the compute?
- The **oracle** row is the ceiling; the gap between the router and the oracle is the room left for better routing.
- A router is only useful if it beats **logistic regression and kNN** on the same features; McNemar p < 0.05 says the accuracy difference is unlikely to be chance.
- **No-gate ablation** shows whether the feature gate earns its parameters.

Outputs in `/kaggle/working`: `results_test.csv`, `results.json`, `pareto_test.png`, `router_release/` (weights + config for Hugging Face)."""),
]


def build(cells) -> dict:
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": src.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, src in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "notebooks"
    for name, cells in [("01_build_oracle_dataset.ipynb", NB01), ("02_train_router.ipynb", NB02)]:
        out = root / name
        out.write_text(json.dumps(build(cells), indent=1, ensure_ascii=False) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()
