"""Router training.

Two objectives share the same ``DynamicNeuralRouter`` network:

* ``"oracle"``      - softmax cross-entropy on the oracle label (which single
  agent is the cheapest correct one). Output: a distribution over agents.
* ``"correctness"`` - independent sigmoid per agent predicting *whether that
  agent will answer correctly* (multi-label BCE on the stored correctness
  matrix). Output: P(correct) per agent, which supports cost-aware decisions
  ("cheapest agent with P(correct) >= tau") and calibration analysis.

Model selection uses validation loss with early stopping; the test split is
never seen during training or selection.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .router import DynamicNeuralRouter


@dataclass
class TrainConfig:
    objective: str = "oracle"  # "oracle" | "correctness"
    hidden: int = 384
    dropout: float = 0.1
    use_gate: bool = True
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 25
    label_smoothing: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss(logits: torch.Tensor, target: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    if cfg.objective == "oracle":
        return F.cross_entropy(logits, target, label_smoothing=cfg.label_smoothing)
    return F.binary_cross_entropy_with_logits(logits, target)


def train_router(
    X_train: np.ndarray,
    T_train: np.ndarray,
    X_val: np.ndarray,
    T_val: np.ndarray,
    num_agents: int,
    cfg: TrainConfig,
    seed: int,
    device: str | None = None,
) -> Tuple[DynamicNeuralRouter, List[Dict[str, float]]]:
    """``T`` is the oracle label vector (objective="oracle") or the n x k
    0/1 correctness matrix (objective="correctness")."""
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    target_dtype = torch.long if cfg.objective == "oracle" else torch.float32
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    Tt = torch.tensor(T_train, dtype=target_dtype, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    Tv = torch.tensor(T_val, dtype=target_dtype, device=device)

    model = DynamicNeuralRouter(Xt.shape[1], num_agents, cfg.hidden, cfg.dropout, cfg.use_gate).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    best_state, best_val, bad, history = None, float("inf"), 0, []
    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen).to(device)
        total = 0.0
        for start in range(0, len(Xt), cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            loss = _loss(model(Xt[idx]), Tt[idx], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = _loss(model(Xv), Tv, cfg).item()
        history.append({"epoch": epoch, "train_loss": total / len(Xt), "val_loss": val})
        if val < best_val - 1e-5:
            best_val, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


@torch.no_grad()
def predict(model: DynamicNeuralRouter, X: np.ndarray, objective: str) -> np.ndarray:
    """Softmax over agents ("oracle") or per-agent sigmoid ("correctness")."""
    device = next(model.parameters()).device
    logits = model(torch.tensor(X, dtype=torch.float32, device=device))
    probs = torch.softmax(logits, -1) if objective == "oracle" else torch.sigmoid(logits)
    return probs.cpu().numpy()
