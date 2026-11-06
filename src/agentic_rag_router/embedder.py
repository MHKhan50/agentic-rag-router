"""Multilingual E5 text embeddings (Wang et al., 2024).

The same 768-d encoder serves two roles: dense retrieval for the RAG agents,
and the input representation for the neural router. E5 requires the
``"query: "`` / ``"passage: "`` prefixes and mean pooling over the attention
mask; both are applied here so callers pass raw text.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class E5Embedder:
    def __init__(self, model_id: str, device: str | None = None, max_length: int = 512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device).eval()
        self.max_length = max_length

    @torch.inference_mode()
    def _encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[i : i + batch_size],
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            out.append(F.normalize(pooled.float(), p=2, dim=1).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 768), np.float32)

    def encode_queries(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        return self._encode([f"query: {t}" for t in texts], batch_size)

    def encode_passages(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        return self._encode([f"passage: {t}" for t in texts], batch_size)

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
