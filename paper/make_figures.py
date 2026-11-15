"""Draws the paper figure from the saved test-split tables in results/ (no model or GPU needed).

  python paper/make_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R02 = ROOT / "results" / "02_train_router"
R03 = ROOT / "results" / "03_answer_cascades"
OUT = Path(__file__).resolve().parent / "figures" / "fig_policies.png"
BLUE, YELLOW, RED, GREEN, GREY = "#1F4E9C", "#E3A800", "#C8102E", "#5FA65A", "#7f7f7f"


def rows(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return {r["policy"]: r for r in csv.DictReader(f)}


def point(r: dict) -> tuple:
    return float(r["gflops/query"]) / 1000.0, 100 * float(r["accuracy"])


def main() -> None:
    q, c = rows(R02 / "results_test.csv"), rows(R03 / "cascade_results_test.csv")
    hull = json.loads((R02 / "results.json").read_text())["fixed_mix_frontier"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    ax.plot([h[0] / 1000 for h in hull], [100 * h[1] for h in hull], "--", color=GREEN, lw=2,
            label="fixed-agent mix (no router)", zorder=1)
    for name in ["A_direct_small", "B_direct_large", "C_rag_small", "D_rag_large", "E_iterative_large"]:
        x, y = point(q["always " + name])
        ax.scatter(x, y, marker="s", s=30, color="black", zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, -11), fontsize=7.5)
    question_only = [("router: correctness @ tau*", BLUE, "neural router, correctness"),
                     ("router: oracle @ tau*", BLUE, None),
                     ("kNN (k=15) @ tau*", YELLOW, "kNN"),
                     ("logistic regression, correctness @ tau*", RED, "logistic regression")]
    for key, col, lab in question_only:
        x, y = point(q[key])
        ax.scatter(x, y, marker="o", s=42, color=col, edgecolor="white", lw=0.6, zorder=4,
                   label=("question-only: " + lab) if lab else None)
    cascades = [("learned, A+C signals (else E) @ tau*", BLUE, "cascade: learned A+C, else E"),
                ("agreement rule (A+C, else E) @ tau*", YELLOW, "cascade: agreement rule, else E"),
                ("learned, C signals only (else E) @ tau*", RED, "cascade: learned C only, else E"),
                ("learned, A+C signals (else D) @ tau*", GREY, "cascade: learned A+C, else D")]
    for key, col, lab in cascades:
        x, y = point(c[key])
        ax.scatter(x, y, marker="^", s=52, color=col, edgecolor="black", lw=0.5, zorder=4, label=lab)
    x, y = point(c["perfect accept (upper bound) @ tau*"])
    ax.scatter(x, y, marker="D", s=40, color="white", edgecolor="black", lw=1.2, zorder=4,
               label="cascade ceiling (C kept exactly when right)")
    x, y = point(q["oracle (upper bound)"])
    ax.scatter(x, y, marker="*", s=170, color="black", zorder=4, label="oracle router")
    ax.set_xlabel("mean compute per question (TFLOPs)")
    ax.set_ylabel("answer accuracy (%)")
    ax.set_xlim(-0.5, 21)
    ax.set_ylim(10, 65)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.8, loc="lower right", frameon=True)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
