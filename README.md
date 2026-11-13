# Agentic RAG Router

**Cost-aware neural routing of queries to specialist retrieval-augmented agents.**

A multi-agent RAG system wastes compute when every query goes to its most
expensive pipeline, and loses accuracy when hard queries go to a cheap one.
This project learns a router that reads a query and sends it to the
*cheapest agent that can still answer it correctly*, and measures that
trade-off with real models, real benchmarks and full cost accounting.

> **Status:** research in progress. The oracle dataset pipeline, router training,
> baselines and statistical evaluation are implemented and tested on CPU. The
> full GPU run has not been done yet, so no results are reported.

## Research question

Can a lightweight neural router over sentence embeddings close most of the
accuracy gap between the cheapest and the strongest RAG agent, at a fraction
of the strongest agent's compute, and does it stay reliable on code-mixed
(English / Roman Urdu) queries?

## Agents

| Agent | Generator (Apache-2.0) | Strategy |
|---|---|---|
| `A_direct_small` | Qwen2.5-1.5B-Instruct | answer from parametric knowledge |
| `B_direct_large` | Qwen2.5-7B-Instruct | answer from parametric knowledge |
| `C_rag_small` | Qwen2.5-1.5B-Instruct | dense retrieval once (top-5), answer |
| `D_rag_large` | Qwen2.5-7B-Instruct | dense retrieval once (top-5), answer |
| `E_iterative_large` | Qwen2.5-7B-Instruct | up to 3 interleaved reason / retrieve steps; repeated queries end the search |

Model size × retrieval form a 2 × 2 grid (A–D), with iterative retrieval (E)
on top, so the results separate what a larger model buys from what retrieval
buys.

Retriever and router features: `intfloat/multilingual-e5-base` (768-d,
94 languages). All agents share one prompt format and greedy decoding, so
differences come from the model and the retrieval strategy only.

## Benchmark questions

| Source | Split | Role |
|---|---|---|
| NQ-Open | validation | open-domain, often answerable without retrieval |
| SQuAD v1.1 | validation | single-hop, needs evidence |
| HotpotQA (distractor) | validation | 2-hop |
| MuSiQue-Ans | validation (answerable) | 2–4 hop connected reasoning |

All paragraphs are pooled into **one shared corpus**, so retrieval faces
cross-question and cross-dataset distractors rather than a per-question
oracle context.

## Oracle labels

Each question is answered by every agent. A prediction is correct when it is
an exact match, contains the gold answer, or reaches token F1 ≥ 0.6 (SQuAD
normalisation, with number words mapped to digits). The threshold is above
0.5 because half-overlap answers ("Articles 101(3) and 102" for "Articles 106
and 107") were wrong in every case inspected.
The label is the cheapest correct agent, with agents ranked by **measured**
mean FLOPs (≈ 2 × parameters × tokens processed). Questions no agent answers
are labelled with the strongest agent and flagged. Every per-agent score and
cost is kept, so labels can be rebuilt for other thresholds or cost weights
without re-running any model. Splits are 70/15/15, stratified by source.

## Repository layout

```
src/agentic_rag_router/
  config.py        experiment configuration (single source of truth)
  data.py          dataset loading, sampling, pooled corpus
  embedder.py      multilingual E5 encoder
  retrieval.py     exact dense retrieval
  llm.py           batched generation with token/time accounting
  prompts.py       shared prompt templates and output parsing
  agents.py        the four agent strategies
  pipeline.py      resumable run of all agents
  metrics.py       EM / F1 / containment
  oracle.py        oracle labels, splits, summary tables
  router.py        gated MLP router
  train.py         router training (oracle and correctness objectives)
  baselines.py     fixed, random, kNN and logistic-regression routers
  evaluation.py    accuracy/cost, Pareto AUC, bootstrap CI, McNemar, ECE
  orchestrator.py  inference-time routing with confidence escalation
notebooks/
  01_build_oracle_dataset.ipynb
  02_train_router.ipynb
scripts/make_notebook.py
tests/test_core.py
```

## Reproduce

**Kaggle (recommended).** Import `notebooks/01_build_oracle_dataset.ipynb`
from this repository, set *Accelerator: GPU T4 x2* and *Internet: On*, run
with `RUN_MODE = "smoke"`, check the tables, then set `RUN_MODE = "full"`.
The full run (2,000 questions) needs about 13–18 GPU hours on T4 x2, so it spans two Kaggle sessions. Start it with *Save Version → Save & Run All (Commit)*. After 11 hours the pipeline saves its work and stops with a `TimeBudgetReached` message; attach that version's output as an input (*Add Input → Your Work*) and run it again, and it continues where it stopped. The 7B model is spread over both GPUs with at most 60% of each GPU's memory used for weights, and a batch that still runs out of memory is retried as two half batches. Then run
`notebooks/02_train_router.ipynb` in the same session (or attach notebook 01's
output as an input) to train the router and all baselines.

**Tests (CPU, no downloads).**

```bash
pip install numpy pandas pytest
pytest -q
```

## Roadmap

1. Oracle dataset (English) — implemented
2. Router training with fixed, random, kNN and logistic-regression baselines,
   Pareto AUC, 5 seeds, bootstrap CIs, McNemar, ECE, gate ablation — implemented
3. Code-mixed English / Roman Urdu query set with human-checked sample
4. Further baselines: zero-shot LLM router, complexity classifier
5. Public release of dataset, router weights and leaderboard

## Licence

Code and router weights: [Apache License 2.0](LICENSE) (see [NOTICE](NOTICE)).
Derived datasets: CC BY-SA 4.0, inherited from the sources — NQ-Open
(CC BY-SA 3.0), SQuAD v1.1 (CC BY-SA 4.0), HotpotQA (CC BY-SA 4.0),
MuSiQue (CC BY 4.0).

## Author

Muzammal Hussain — Software Engineer, Data Scientist and AI Researcher
· ORCID [0009-0001-2570-5686](https://orcid.org/0009-0001-2570-5686)
