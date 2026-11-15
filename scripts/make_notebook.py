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
- learned baselines on the same 768-d embeddings: cosine kNN and logistic regression, each trained both on the oracle label and on per-agent correctness
- the fixed-agent mix: the best accuracy reachable at each compute by sending a fixed share of questions to each agent, with no router
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
# Same task as the correctness router (per-agent P(correct)), so all three are scored with the same policy.
probs["kNN (k=15), correctness"] = [(bl.knn_correctness(X[tr], C[tr], X[va]), bl.knn_correctness(X[tr], C[tr], X[te]))]
lrc = bl.fit_logistic_correctness(X[tr], C[tr])
probs["logistic regression, correctness"] = [(lrc(X[va]), lrc(X[te]))]
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

FIXED_MIX = ev.fixed_mix_frontier(C[te], FL[te])   # best accuracy reachable at each compute without a router
rows = []
for name, cs in choices.items():
    ms = [ev.policy_metrics(c, C[te], FL[te], y[te]) for c in cs]
    gf = np.mean([m["gflops"] for m in ms])
    acc = np.array([m["accuracy"] for m in ms])
    lo, hi = ev.bootstrap_ci(C[te][np.arange(len(te)), cs[0]].astype(float))
    rows.append({
        "policy": name, "seeds": len(ms),
        "accuracy": acc.mean(), "acc_std": acc.std(), "acc_ci95": f"[{lo:.3f}, {hi:.3f}]",
        "gflops/query": np.mean([m["gflops"] for m in ms]),
        "cost_vs_strongest": np.mean([m["cost_vs_strongest"] for m in ms]),
        "routing_acc": np.mean([m["routing_accuracy"] for m in ms]),
        "fixed_mix_acc_same_cost": ev.frontier_accuracy(FIXED_MIX, gf),
        "gain_vs_fixed_mix": acc.mean() - ev.frontier_accuracy(FIXED_MIX, gf),
    })
results = pd.DataFrame(rows).set_index("policy").round(4)
print("Test split. CI = 95% bootstrap over questions (seed 0). cost_vs_strongest = compute relative to always using the strongest agent.")\nprint("gain_vs_fixed_mix = accuracy minus the best accuracy reachable at the same compute by mixing fixed agents (no router).")
display(results)
print("tau* chosen on validation:", {k: [round(t, 3) for t in v] for k, v in tau_star.items()})"""),
    ("code", """import matplotlib.pyplot as plt

# Every learned policy is swept over its threshold. Curves are compared only on the compute range that all of
# them (and the fixed-agent mix) cover, so no curve is credited or penalised for a range it cannot reach.
max_cost = FL[te][:, -1].mean() / 1e9
sweeps = {name: [[(c, a) for _, c, a in ev.sweep(policy_fn(name), pt, C[te], FL[te], TAUS)] for _, pt in runs]
          for name, runs in probs.items()}
lo = max([FIXED_MIX[0][0]] + [min(c for c, _ in pts) for runs in sweeps.values() for pts in runs])
hi = max_cost
area = {name: (float(np.mean([ev.area_on_range(p, lo, hi) for p in runs])),
               float(np.std([ev.area_on_range(p, lo, hi) for p in runs]))) for name, runs in sweeps.items()}
area["fixed-agent mix (no router)"] = (ev.frontier_area_on_range(FIXED_MIX, lo, hi), 0.0)
print(f"mean accuracy over the shared compute range {lo:.0f}-{hi:.0f} GFLOPs (mean ± std over seeds):")
display(pd.Series({k: f"{m:.4f} ± {s:.4f}" for k, (m, s) in sorted(area.items(), key=lambda kv: -kv[1][0])}, name="mean accuracy"))

BLUE, YELLOW, RED, GREEN, GREY = "#1F4E9C", "#E3A800", "#C8102E", "#5FA65A", "#7f7f7f"
show = {"router: correctness": BLUE, "kNN (k=15), correctness": YELLOW,
        "logistic regression, correctness": RED, "router: oracle": GREY}
fig, ax = plt.subplots(figsize=(7.5, 5))
xs, ys = zip(*FIXED_MIX)
ax.plot(xs, ys, "--", color=GREEN, lw=2, label="fixed-agent mix (no router)")
for name, col in show.items():
    cs, accs = zip(*ev.pareto_front(sweeps[name][0]))
    ax.step(cs, accs, where="post", color=col, lw=2, label=name)
for k, a in enumerate(ORDER):
    m = ev.policy_metrics(bl.fixed(len(te), k), C[te], FL[te])
    ax.scatter(m["gflops"], m["accuracy"], marker="s", s=36, color="black", zorder=3)
    ax.annotate(a, (m["gflops"], m["accuracy"]), textcoords="offset points", xytext=(5, -10), fontsize=8)
m = ev.policy_metrics(y[te], C[te], FL[te])
ax.scatter(m["gflops"], m["accuracy"], marker="*", s=160, color="black", zorder=4, label="oracle router (upper bound)")
ax.axvspan(lo, hi, color=GREEN, alpha=0.06)
ax.set_xlabel("mean GFLOPs per query"); ax.set_ylabel("answer accuracy (correct)")
ax.set_title("Accuracy vs compute on the test split"); ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(work / "pareto_test.png", dpi=200); plt.show()"""),
    ("code", """def hits(policy, seed=0):
    return C[te][np.arange(len(te)), choices[policy][seed]]

strongest = f"always {ORDER[-1]}"
tests = {}
for name in ["router: oracle @ tau*", "router: correctness @ tau*", "router: correctness (default)"]:
    for ref in [strongest, "logistic regression, correctness (default)", "kNN (k=15), correctness (default)"]:
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
        vmix = ev.fixed_mix_frontier(C[va], FL[va])
        a = ev.area_on_range([(c, acc) for _, c, acc in pts], max(vmix[0][0], min(c for _, c, _ in pts)), FL[va][:, -1].mean() / 1e9)
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
    "tau": tau_star[name][seed], "val_mean_accuracy_over_range": val_auc,
    "data_commit": data_manifest["repo_commit"], "code_commit": COMMIT,
}, indent=2))
results.to_csv(work / "results_test.csv")
(work / "results.json").write_text(json.dumps({
    "mean_accuracy_shared_range": area, "shared_range_gflops": [lo, hi], "fixed_mix_frontier": FIXED_MIX, "tau_star": tau_star, "mcnemar": tests, "ece_correctness": ece,
    "splits": {k: int(len(v)) for k, v in S.items()}, "data_dir": str(DATA_DIR),
}, indent=2, default=float))
print(f"exported {name} (seed {seed}, validation mean accuracy over range {val_auc:.4f}) ->", EXPORT)"""),
    ("markdown", """### Reading the results
- **gain_vs_fixed_mix** is the headline: how much more accurate is a policy than simply sending a fixed share of questions to each agent at the same compute? A router earns its place only where this is clearly above zero.
- **accuracy / gflops** of `router … @ tau*` against `always <strongest>`: same accuracy (within 1 point, by construction on validation) at what fraction of the compute?
- The **oracle** row is the ceiling; the gap between the router and the oracle is the room left for better routing.
- A router is only useful if it beats **logistic regression and kNN** trained for the same task on the same features; the shared-range mean accuracy compares whole curves fairly, and McNemar p < 0.05 says an accuracy difference at one operating point is unlikely to be chance.
- **No-gate ablation** shows whether the feature gate earns its parameters.

Outputs in `/kaggle/working`: `results_test.csv`, `results.json`, `pareto_test.png`, `router_release/` (weights + config for Hugging Face)."""),
]

NB03 = [
    ("markdown", """# Agentic RAG Router — 03: Answer-level cascades

Notebook 02 showed that routers which see **only the question** (the neural router, kNN, logistic regression) stay within about one point of a fixed mix of agents, while the oracle reaches 0.61. This notebook asks whether looking at **cheap agents' answers** closes more of that gap.

A cascade runs one or two cheap agents first, then keeps the small RAG agent's answer or escalates to a stronger agent. Every agent that runs is paid for. Signals (all available at inference time, no gold answers or gold passages):

- agreement between `A_direct_small` and `C_rag_small` (token F1 and exact match after SQuAD normalisation)
- answer length and a refusal flag ("unknown", "not mentioned", empty)
- cosine similarity of the best passage retrieved by `C_rag_small`

| Cascade | Runs first | Keeps | Escalates to | Score |
|---|---|---|---|---|
| agreement rule | A, C | C | E | A–C token F1 |
| learned, A+C signals | A, C | C | E | logistic regression on all signals |
| learned, C signals only | C | C | E | logistic regression on C's length, refusal, retrieval score |
| learned, escalate to D | A, C | C | D | as the A+C model |
| perfect accept (upper bound) | C | C | E | 1 if C is correct |

τ* is chosen on the validation split with the notebook 02 rule (cheapest setting within 1 point of the strongest agent), then applied unchanged to the test split. The question-only logistic-regression router from notebook 02 is retrained here as the reference.

No GPU needed (Internet On, to fetch the code): everything is computed from notebook 01's saved answers. **Input:** add notebook 01's output (*Add Input → Your Work → Agentic RAG Router – Oracle Dataset*)."""),
    ("code", """import os, sys, subprocess, json, glob, platform
from pathlib import Path

REPO_URL = "https://github.com/MHKhan50/agentic-rag-router"
REPO_DIR = Path("/kaggle/working/agentic-rag-router") if Path("/kaggle").exists() else Path("agentic-rag-router").resolve()
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
else:
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
sys.path.insert(0, str(REPO_DIR / "src"))
COMMIT = subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

import numpy as np, pandas as pd, sklearn
from agentic_rag_router import baselines as bl, evaluation as ev, cascade as cs
print("python", platform.python_version(), "| sklearn", sklearn.__version__, "| repo commit", COMMIT)

work = Path("/kaggle/working" if Path("/kaggle").exists() else ".")
cands = sorted(glob.glob("/kaggle/input/**/oracle_labels.csv", recursive=True)) + sorted(str(p) for p in work.glob("oracle_*/oracle_labels.csv"))
assert cands, "No oracle_labels.csv found: add notebook 01's output as an input."
DATA_DIR = Path(([c for c in cands if "oracle_full" in c] or cands)[0]).parent
print("using", DATA_DIR)

oracle = pd.read_csv(DATA_DIR / "oracle_labels.csv")
data_manifest = json.loads((DATA_DIR / "manifest.json").read_text())
ORDER = data_manifest["agent_cost_order"]
qids = json.loads((DATA_DIR / "query_emb_qids.json").read_text())
emb = np.load(DATA_DIR / "query_emb.npy")
pos = {q: i for i, q in enumerate(qids)}
X = emb[[pos[q] for q in oracle["qid"]]].astype(np.float32)
C = oracle[[f"correct__{a}" for a in ORDER]].to_numpy(dtype=int)
FL = oracle[[f"flops__{a}" for a in ORDER]].to_numpy(dtype=float)
S = {s: np.where(oracle["split"].to_numpy() == s)[0] for s in ["train", "val", "test"]}
tr, va, te = S["train"], S["val"], S["test"]
print({s: len(i) for s, i in S.items()}, "| agents (cheap -> strong):", ORDER)
if len(tr) < 500:
    print("WARNING: smoke-sized data. This run only checks the code path; the numbers are not results.")"""),
    ("code", """A, B, Cr, D, E = ORDER   # A_direct_small, B_direct_large, C_rag_small, D_rag_large, E_iterative_large
long = pd.read_csv(DATA_DIR / "agent_results_long.csv", keep_default_na=False)
preds = cs.predictions_wide(long, oracle["qid"], ORDER)

# Best-passage similarity for the single-retrieval agents (query and passage embeddings from notebook 01).
corpus_pids = [json.loads(l)["pid"] for l in open(DATA_DIR / "corpus.jsonl", encoding="utf-8")]
P = np.load(DATA_DIR / "passage_emb.npy")
assert len(corpus_pids) == len(P), "corpus.jsonl and passage_emb.npy are not aligned"
pid_index = {p: i for i, p in enumerate(corpus_pids)}
retrieval = {}
for agent in (Cr, D):
    runs = {}
    for l in open(DATA_DIR / "runs" / f"{agent}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        runs[r["qid"]] = r.get("retrieved_pids", [])
    retrieval[agent] = cs.top_retrieval_score(X, P, pid_index, [runs.get(q, []) for q in oracle["qid"]])
F = cs.answer_signals(preds, retrieval)
F["refusal__any_AC"] = np.maximum(F[f"refusal__{A}"], F[f"refusal__{Cr}"])
print(F.shape, "signals; retrieval score missing for", int(np.isnan(F[f"retr_top__{Cr}"]).sum()), "questions")
cC = ORDER.index(Cr)
print("C_rag_small correct on train:", C[tr, cC].mean().round(3))
display(pd.DataFrame({
    "C correct when A and C agree (exact)": [C[tr, cC][F[f"agree_em__{A}__{Cr}"].to_numpy()[tr] == 1].mean()],
    "C correct when they disagree": [C[tr, cC][F[f"agree_em__{A}__{Cr}"].to_numpy()[tr] == 0].mean()],
    "share agreeing (train)": [F[f"agree_em__{A}__{Cr}"].to_numpy()[tr].mean()],
}).round(3))"""),
    ("code", """AC_FEATS = [f"len__{A}", f"len__{Cr}", f"refusal__{A}", f"refusal__{Cr}", f"agree_f1__{A}__{Cr}", f"agree_em__{A}__{Cr}", f"retr_top__{Cr}"]
C_FEATS = [f"len__{Cr}", f"refusal__{Cr}", f"retr_top__{Cr}"]
CASCADES = {
    "agreement rule (A+C, else E)": cs.Cascade("agreement rule", (A, Cr), Cr, E),
    "learned, A+C signals (else E)": cs.Cascade("learned A+C", (A, Cr), Cr, E, tuple(AC_FEATS)),
    "learned, C signals only (else E)": cs.Cascade("learned C", (Cr,), Cr, E, tuple(C_FEATS)),
    "learned, A+C signals (else D)": cs.Cascade("learned A+C to D", (A, Cr), Cr, D, tuple(AC_FEATS)),
    "perfect accept (upper bound)": cs.Cascade("perfect accept", (Cr,), Cr, E),
}
scores = {}
for name, c in CASCADES.items():
    if name.startswith("agreement"):
        s = F[f"agree_f1__{A}__{Cr}"].to_numpy()
    elif name.startswith("perfect"):
        s = C[:, cC].astype(float)
    else:
        f = cs.fit_accept_model(F.loc[:, list(c.features)].to_numpy()[tr], C[tr, cC], seed=0)
        s = f(F.loc[:, list(c.features)].to_numpy())
    scores[name] = s
TAUS = np.linspace(0, 1.0001, 41)
tau_star = {n: cs.pick_tau(c, scores[n][va], C[va], FL[va], ORDER, TAUS) for n, c in CASCADES.items()}
print("tau* (validation):", {k: round(v, 3) for k, v in tau_star.items()})"""),
    ("code", """FIXED_MIX = ev.fixed_mix_frontier(C[te], FL[te])
strongest_hit = C[te][:, -1]
rows, hits = [], {}
def add_row(name, hit, cost, esc=None):
    g = cost.mean() / 1e9
    lo, hi = ev.bootstrap_ci(hit.astype(float))
    mc = ev.mcnemar_exact(hit, strongest_hit)
    rows.append({"policy": name, "accuracy": hit.mean(), "acc_ci95": f"[{lo:.3f}, {hi:.3f}]", "gflops/query": g,
                 "cost_vs_strongest": cost.sum() / FL[te][:, -1].sum(), "escalation_rate": esc,
                 "fixed_mix_acc_same_cost": ev.frontier_accuracy(FIXED_MIX, g),
                 "gain_vs_fixed_mix": hit.mean() - ev.frontier_accuracy(FIXED_MIX, g),
                 "vs_strongest_only_this": mc["a_only"], "vs_strongest_only_strongest": mc["b_only"], "p_mcnemar_vs_strongest": mc["p_value"]})
    hits[name] = hit

for k, a in enumerate(ORDER):
    add_row(f"always {a}", C[te][:, k], FL[te][:, k])
add_row("oracle router (upper bound)", C[te].max(axis=1), FL[te][np.arange(len(te)), oracle["oracle_label"].to_numpy()[te]])
for name, c in CASCADES.items():
    for label, t in (("@ tau*", tau_star[name]),) + ((("@ 0.5", 0.5),) if not name.startswith("perfect") else ()):
        accept = scores[name][te] >= t
        hit, cost = cs.cascade_outcome(c, accept, C[te], FL[te], ORDER)
        add_row(f"{name} {label}", hit, cost, 1 - accept.mean())

# Question-only reference from notebook 02: logistic regression estimating P(correct) for every agent.
lrc = bl.fit_logistic_correctness(X[tr], C[tr])
pv, pt = lrc(X[va]), lrc(X[te])
TAUS_Q = np.linspace(0, 1, 41)
target = C[va][:, -1].mean() - 0.01
stats = [(float(t), ev.policy_metrics(ev.cheapest_confident(pv, float(t)), C[va], FL[va])) for t in TAUS_Q]
ok = [(t, m["gflops"]) for t, m in stats if m["accuracy"] >= target]
tq = min(ok, key=lambda x: x[1])[0] if ok else max(stats, key=lambda x: x[1]["accuracy"])[0]
ch = ev.cheapest_confident(pt, tq)
add_row("question-only logistic regression @ tau*", C[te][np.arange(len(te)), ch], FL[te][np.arange(len(te)), ch])
tau_star["question-only logistic regression"] = tq

results = pd.DataFrame(rows).set_index("policy").round(4)
print("Test split (", len(te), "questions). CI = 95% bootstrap over questions. McNemar against always", ORDER[-1])
display(results)"""),
    ("code", """import matplotlib.pyplot as plt

curves = {n: [(g, a) for _, g, a in cs.cascade_sweep(c, scores[n][te], C[te], FL[te], ORDER, TAUS)]
          for n, c in CASCADES.items() if not n.startswith("perfect")}
curves["question-only logistic regression"] = [(g, a) for _, g, a in ev.sweep(ev.cheapest_confident, pt, C[te], FL[te], TAUS_Q)]
lo = max([FIXED_MIX[0][0]] + [min(g for g, _ in p) for p in curves.values()])
hi = FL[te][:, -1].mean() / 1e9
area = {n: ev.area_on_range(p, lo, hi) for n, p in curves.items()}
area["fixed-agent mix (no router)"] = ev.frontier_area_on_range(FIXED_MIX, lo, hi)
print(f"mean accuracy over the shared compute range {lo:.0f}-{hi:.0f} GFLOPs:")
display(pd.Series(area, name="mean accuracy").sort_values(ascending=False).round(4))
best_gain = {n: max(a - ev.frontier_accuracy(FIXED_MIX, g) for g, a in p) for n, p in curves.items()}
display(pd.Series(best_gain, name="largest gain over the fixed mix on the curve").sort_values(ascending=False).round(4))

BLUE, YELLOW, RED, GREEN, GREY = "#1F4E9C", "#E3A800", "#C8102E", "#5FA65A", "#7f7f7f"
show = {"learned, A+C signals (else E)": BLUE, "agreement rule (A+C, else E)": YELLOW,
        "learned, C signals only (else E)": RED, "question-only logistic regression": GREY}
fig, ax = plt.subplots(figsize=(7.5, 5))
xs, ys = zip(*FIXED_MIX)
ax.plot(xs, ys, "--", color=GREEN, lw=2, label="fixed-agent mix (no router)")
for n, col in show.items():
    c_, a_ = zip(*ev.pareto_front(curves[n]))
    ax.step(c_, a_, where="post", color=col, lw=2, label=n)
for k, a in enumerate(ORDER):
    ax.scatter(FL[te][:, k].mean() / 1e9, C[te][:, k].mean(), marker="s", s=36, color="black", zorder=3)
    ax.annotate(a, (FL[te][:, k].mean() / 1e9, C[te][:, k].mean()), textcoords="offset points", xytext=(5, -10), fontsize=8)
r = results.loc["oracle router (upper bound)"]
ax.scatter(r["gflops/query"], r["accuracy"], marker="*", s=160, color="black", zorder=4, label="oracle router (upper bound)")
ax.set_xlabel("mean GFLOPs per query"); ax.set_ylabel("answer accuracy (correct)")
ax.set_title("Answer-level cascades vs question-only routing (test split)"); ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(work / "cascade_pareto.png", dpi=200); plt.show()"""),
    ("code", """# Where does the best learned cascade at tau* help or hurt? Accuracy by source dataset.
best = "learned, A+C signals (else E)"
accept = scores[best][te] >= tau_star[best]
hit, cost = cs.cascade_outcome(CASCADES[best], accept, C[te], FL[te], ORDER)
ds = oracle["dataset"].to_numpy()[te]
by_ds = pd.DataFrame({"cascade_acc": pd.Series(hit).groupby(ds).mean(), f"always_{E}_acc": pd.Series(strongest_hit).groupby(ds).mean(),
                      "escalation_rate": pd.Series(1 - accept.astype(float)).groupby(ds).mean(),
                      "cost_vs_strongest": pd.Series(cost).groupby(ds).sum() / pd.Series(FL[te][:, -1]).groupby(ds).sum(),
                      "questions": pd.Series(hit).groupby(ds).size()}).round(3)
display(by_ds)

results.to_csv(work / "cascade_results_test.csv")
(work / "cascade_results.json").write_text(json.dumps({
    "tau_star": tau_star, "shared_range_gflops": [lo, hi], "mean_accuracy_shared_range": area,
    "largest_gain_over_fixed_mix": best_gain, "by_dataset": by_ds.reset_index().to_dict("records"),
    "features": {"A+C": AC_FEATS, "C": C_FEATS}, "splits": {k: int(len(v)) for k, v in S.items()},
    "data_commit": data_manifest.get("repo_commit"), "code_commit": COMMIT, "data_dir": str(DATA_DIR),
}, indent=2, default=float))
print("saved cascade_results_test.csv, cascade_results.json, cascade_pareto.png in", work)"""),
    ("markdown", """### Reading the results
- **gain_vs_fixed_mix** at τ*: positive means the cascade is more accurate than any fixed share of agents at the same compute.
- Compare the cascades with **question-only logistic regression** (the notebook 02 reference): the difference is what looking at cheap answers adds.
- **perfect accept** shows the ceiling of this two-step design (C kept exactly when it is right); the gap between it and the learned cascades is what better confidence signals could still win.
- McNemar against the strongest agent: p ≥ 0.05 means no detectable accuracy difference at that operating point, which is not proof that the two are equal."""),
]


def build(cells, accelerator: str = "nvidiaTeslaT4", internet: bool = True) -> dict:
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
            "kaggle": {"accelerator": accelerator, "isInternetEnabled": internet},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "notebooks"
    for name, cells, acc in [("01_build_oracle_dataset.ipynb", NB01, "nvidiaTeslaT4"),
                             ("02_train_router.ipynb", NB02, "nvidiaTeslaT4"),
                             ("03_answer_cascades.ipynb", NB03, "none")]:
        out = root / name
        out.write_text(json.dumps(build(cells, acc), indent=1, ensure_ascii=False) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()
