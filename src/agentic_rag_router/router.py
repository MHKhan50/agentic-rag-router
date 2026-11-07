"""Neural router: query embedding -> distribution over specialist agents."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicNeuralRouter(nn.Module):
    """Gated MLP over a frozen sentence embedding.

    A sigmoid feature gate rescales each input dimension per query, letting
    the router emphasise the embedding directions that matter for routing
    before a LayerNorm-GELU-Dropout projection to agent logits.

    ``forward`` returns logits (train with ``F.cross_entropy``);
    ``predict_proba`` returns probabilities for inference. ``mask`` (1 = agent
    available, 0 = unavailable) removes agents from consideration, e.g. when
    one is down or over budget.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_agents: int = 4,
        hidden: int = 384,
        dropout: float = 0.1,
        use_gate: bool = True,
    ):
        super().__init__()
        self.gate = nn.Linear(d_model, d_model) if use_gate else None
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_agents),
        )

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.gate is not None:
            embeddings = embeddings * torch.sigmoid(self.gate(embeddings))
        logits = self.mlp(embeddings)
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=logits.device)
            if not bool(mask.any(dim=-1).all()):
                raise ValueError("every row of mask must allow at least one agent")
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits

    @torch.no_grad()
    def predict_proba(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.softmax(self.forward(embeddings, mask), dim=-1)
