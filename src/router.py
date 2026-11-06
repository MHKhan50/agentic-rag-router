import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DynamicNeuralRouter(nn.Module):
    def __init__(self, d_model: int = 768, num_agents: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feature_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_agents)
        )

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.feature_gate(embeddings)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float(\"-inf\"))
        return F.softmax(logits, dim=-1)
