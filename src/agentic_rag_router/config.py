"""Experiment configuration.

All knobs that affect results live here so that a run is fully described by
one serialisable object (written to ``manifest.json`` next to the outputs).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

# Both generator models are Apache-2.0 licensed, so the agents, the oracle
# labels they produce and any router trained on them can be released and used
# commercially. (Qwen2.5-3B-Instruct is deliberately avoided: it ships under
# the non-commercial "qwen-research" licence.)
SMALL_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
LARGE_LLM = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"  # 768-d, 94 languages, MIT


@dataclass(frozen=True)
class AgentSpec:
    """One specialist agent = a generator model + a retrieval strategy."""

    name: str
    model_id: str
    strategy: str  # "direct" | "single_rag" | "iterative_rag"
    top_k: int = 0  # passages retrieved per retrieval call
    max_steps: int = 1  # reasoning/retrieval rounds (iterative only)
    max_passages: int = 0  # cap on passages placed in the prompt
    batch_size: int = 16


# A 2 x 2 design (model size x retrieval) plus iterative retrieval, so results
# separate what a bigger model buys from what retrieval buys.
DEFAULT_AGENTS: List[AgentSpec] = [
    AgentSpec("A_direct_small", SMALL_LLM, "direct", batch_size=64),
    AgentSpec("B_direct_large", LARGE_LLM, "direct", batch_size=32),
    AgentSpec("C_rag_small", SMALL_LLM, "single_rag", top_k=5, max_passages=5, batch_size=32),
    AgentSpec("D_rag_large", LARGE_LLM, "single_rag", top_k=5, max_passages=5, batch_size=16),
    AgentSpec(
        "E_iterative_large",
        LARGE_LLM,
        "iterative_rag",
        top_k=3,
        max_steps=3,
        max_passages=9,
        batch_size=16,
    ),
]


@dataclass
class PipelineConfig:
    seed: int = 42
    # Questions sampled per source dataset (validation splits).
    n_per_dataset: Dict[str, int] = field(
        default_factory=lambda: {"nq_open": 500, "squad": 500, "hotpotqa": 500, "musique": 500}
    )
    agents: List[AgentSpec] = field(default_factory=lambda: list(DEFAULT_AGENTS))
    embedding_model: str = EMBEDDING_MODEL
    passage_max_words: int = 120  # passage truncation inside prompts
    answer_max_new_tokens: int = 32
    step_max_new_tokens: int = 48
    # Correct = exact match, or gold answer contained in the prediction, or
    # token F1 >= this threshold (see metrics.is_correct).
    correct_f1_threshold: float = 0.6
    split_fractions: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def smoke(cls) -> "PipelineConfig":
        """Tiny run (~10 min on a Kaggle T4x2) to validate the whole pipeline."""
        return cls(n_per_dataset={"nq_open": 10, "squad": 10, "hotpotqa": 10, "musique": 10})
