import logging
import torch
from typing import Any, Dict
from src.router import DynamicNeuralRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentOrchestrator:
    def __init__(self, router_weights_path: str = None, d_model: int = 768):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agents = ["retrieval_agent", "reasoning_agent", "code_agent", "verifier_agent"]
        self.router = DynamicNeuralRouter(d_model=d_model, num_agents=len(self.agents)).to(self.device)

    def route_query(self, query_embedding: torch.Tensor) -> Dict[str, Any]:
        query_embedding = query_embedding.to(self.device)
        with torch.no_grad():
            probs = self.router(query_embedding)
            selected_idx = torch.argmax(probs, dim=-1).item()
        selected_agent = self.agents[selected_idx]
        confidence = probs[0][selected_idx].item()
        return {"target_agent": selected_agent, "confidence": confidence}
