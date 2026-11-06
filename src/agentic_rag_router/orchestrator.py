"""Inference-time orchestration: embed a query, pick an agent, report confidence."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .config import DEFAULT_AGENTS
from .router import DynamicNeuralRouter


class AgentOrchestrator:
    def __init__(
        self,
        agent_names: Optional[List[str]] = None,
        router_weights_path: Optional[str] = None,
        d_model: int = 768,
        min_confidence: float = 0.0,
    ):
        """``min_confidence``: below this the query escalates to the most
        capable agent instead of trusting an unsure routing decision."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agents = agent_names or [a.name for a in DEFAULT_AGENTS]
        self.router = DynamicNeuralRouter(d_model=d_model, num_agents=len(self.agents)).to(self.device)
        if router_weights_path:
            self.router.load_state_dict(torch.load(router_weights_path, map_location=self.device))
        self.router.eval()
        self.min_confidence = min_confidence

    def route(self, query_embedding: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, object]:
        emb = query_embedding.to(self.device)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        probs = self.router.predict_proba(emb, mask)[0]
        idx = int(torch.argmax(probs))
        confidence = float(probs[idx])
        escalated = confidence < self.min_confidence
        if escalated:
            idx = len(self.agents) - 1
        return {
            "target_agent": self.agents[idx],
            "confidence": confidence,
            "escalated": escalated,
            "probabilities": dict(zip(self.agents, probs.tolist())),
        }
