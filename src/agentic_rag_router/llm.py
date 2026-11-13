"""Batched chat generation with per-call token accounting.

``ChatLLM`` is the minimal interface the agents depend on, which keeps agent
logic testable with a fake backend. ``HFChatLLM`` implements it with
Hugging Face ``transformers`` (greedy decoding, so runs are deterministic).
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import List, Protocol, Sequence

Messages = List[dict]


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    gen_tokens: int
    seconds: float  # batch wall-clock time divided evenly across the batch


class ChatLLM(Protocol):
    model_id: str
    n_params: int

    def generate(self, prompts: Sequence[Messages], max_new_tokens: int, batch_size: int) -> List[Generation]:
        ...


class HFChatLLM:
    def __init__(self, model_id: str, device_map: str | dict = "auto", gpu_fraction: float = 0.6):
        """``device_map="auto"`` shards a large model over all visible GPUs;
        pass ``{"": 0}`` to keep a small model on one GPU (faster, no hops).

        When sharding, each GPU receives at most ``gpu_fraction`` of its memory
        for weights. Without this cap ``accelerate`` fills GPU 0 almost
        completely, and long retrieval prompts then run out of memory there."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # T4/P100 have no bfloat16; fall back to float16 there.
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32
        max_memory = None
        if torch.cuda.is_available() and device_map == "auto" and torch.cuda.device_count() > 1:
            max_memory = {
                i: f"{gpu_fraction * torch.cuda.get_device_properties(i).total_memory / 2**30:.1f}GiB"
                for i in range(torch.cuda.device_count())
            }
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map if torch.cuda.is_available() else None,
            max_memory=max_memory,
        ).eval()
        placement = set(getattr(self.model, "hf_device_map", {}).values())
        if placement & {"cpu", "disk"}:
            print(f"WARNING: part of {model_id} was placed on {placement & {'cpu', 'disk'}}; generation will be slow")
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def _render(self, messages: Messages) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts: Sequence[Messages], max_new_tokens: int, batch_size: int) -> List[Generation]:
        texts = [self._render(m) for m in prompts]
        # Length-sorted batching minimises padding; results are restored to input order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        results: List[Generation | None] = [None] * len(texts)
        device = next(self.model.parameters()).device
        for start in range(0, len(order), batch_size):
            self._generate_batch(texts, order[start : start + batch_size], max_new_tokens, device, results)
        return results  # type: ignore[return-value]

    def _generate_batch(self, texts, idx, max_new_tokens, device, results) -> None:
        """Generate one batch; on CUDA out-of-memory, retry it as two half batches."""
        torch = self._torch
        enc = out = None
        oom = False
        try:
            enc = self.tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True).to(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            if len(idx) == 1:
                raise
            oom = True
        if oom:  # outside the except block, so the failed batch's tensors can be freed
            enc = out = None
            gc.collect()
            torch.cuda.empty_cache()
            half = len(idx) // 2
            self._generate_batch(texts, idx[:half], max_new_tokens, device, results)
            self._generate_batch(texts, idx[half:], max_new_tokens, device, results)
            return
        per_item = (time.perf_counter() - t0) / len(idx)
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        for row, i in enumerate(idx):
            toks = new_tokens[row]
            eos = (toks == self.tokenizer.eos_token_id) | (toks == self.tokenizer.pad_token_id)
            n_gen = int(eos.int().argmax()) if bool(eos.any()) else int(toks.numel())
            results[i] = Generation(
                text=self.tokenizer.decode(toks[:n_gen], skip_special_tokens=True).strip(),
                prompt_tokens=int(prompt_lens[row]),
                gen_tokens=n_gen,
                seconds=per_item,
            )

    def close(self) -> None:
        del self.model
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
