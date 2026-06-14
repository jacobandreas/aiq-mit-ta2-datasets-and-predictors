"""Per-item MLP attribution via gradient × activation.

Adapted from arithmetic-inconsistencies/attribution/src/attribution.py.
Uses the same forward_pre_hook idiom but without activation patching:
for each item we run a single forward+backward pass on −log P(y|x) and
record grad ⊙ activation at the last prompt-token position for every
MLP layer.

Output shape per item: [num_layers, intermediate_size]  (float32 numpy)
Output shape for N items: [N, num_layers, intermediate_size]

Supports GPT-2 and Llama/Qwen family models.

Architecture detection
----------------------
  GPT-2  (model.transformer.h)     hooks l.mlp.c_proj
  Llama  (model.model.layers)       hooks l.mlp.down_proj
  Qwen   (model.model.layers)       hooks l.mlp.down_proj
  Mistral / Gemma / Phi             hooks l.mlp.down_proj  (same as Llama)

We hook the INPUT to the final linear in each MLP block because that is the
"intermediate" representation — same convention as the original patching code.
"""
from __future__ import annotations

import gc
from typing import List, Optional, Tuple

import numpy as np
import torch


# ── Architecture utilities ────────────────────────────────────────────────────

def detect_family(model) -> str:
    name = type(model).__name__.lower()
    if "gpt2" in name:
        return "gpt2"
    return "llama"   # covers Llama, Qwen, Mistral, Gemma, Phi …


def build_model_config(model) -> dict:
    """Extract the config dict expected by hook utilities."""
    family = detect_family(model)
    cfg = model.config
    if family == "gpt2":
        n_inner = getattr(cfg, "n_inner", None) or (4 * cfg.n_embd)
        return {
            "family": family,
            "num_layers": cfg.n_layer,
            "intermediate_size": n_inner,
        }
    return {
        "family": family,
        "num_layers": cfg.num_hidden_layers,
        "intermediate_size": cfg.intermediate_size,
    }


def _hook_targets(model, family: str) -> List:
    """Return the list of nn.Module objects to hook (one per layer)."""
    if family == "gpt2":
        return [layer.mlp.c_proj for layer in model.transformer.h]
    return [layer.mlp.down_proj for layer in model.model.layers]


# ── Activation capturer (forward_pre_hook) ───────────────────────────────────

class _Capturer:
    """Registers forward_pre_hooks on a list of modules.

    On each forward call the hook:
      1. detaches + clones the input activation,
      2. optionally enables grad on the clone,
      3. replaces the original input with the clone,
      4. stores the clone in self.captures[layer_idx].

    After a backward call, captures[i].grad contains d(loss)/d(act_i).
    """

    def __init__(self, modules: list, enable_grad: bool):
        self.modules = modules
        self.enable_grad = enable_grad
        self.captures: List[Optional[torch.Tensor]] = [None] * len(modules)
        self._handles: list = []

    def __enter__(self):
        for li, mod in enumerate(self.modules):
            self._handles.append(mod.register_forward_pre_hook(self._make_hook(li)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, li: int):
        def hook(_module, inputs):
            x = inputs[0]
            x_new = x.detach().clone()
            if self.enable_grad:
                x_new.requires_grad_(True)
            self.captures[li] = x_new
            return (x_new,) + tuple(inputs[1:])
        return hook


# ── Tokenisation helpers ──────────────────────────────────────────────────────

def _prepare_inputs(
    items: List[dict],
    tokenizer,
    max_length: int = 512,
) -> Tuple[List[torch.Tensor], List[int]]:
    """Tokenise x+y pairs and return (input_ids_list, prompt_lens).

    The prompt is tokenised with add_special_tokens=True (prepends BOS for
    Llama); the completion is appended without extra specials.
    """
    all_ids, prompt_lens = [], []
    for item in items:
        prompt_ids = tokenizer(item["x"], add_special_tokens=True)["input_ids"]
        answer_ids = tokenizer(item["y"], add_special_tokens=False)["input_ids"]
        full_ids = (prompt_ids + answer_ids)[:max_length]
        all_ids.append(torch.tensor(full_ids, dtype=torch.long))
        prompt_lens.append(len(prompt_ids))
    return all_ids, prompt_lens


def _pad(id_list: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    """Left-pad variable-length sequences to the same length."""
    max_len = max(t.shape[0] for t in id_list)
    out = torch.full((len(id_list), max_len), pad_id, dtype=torch.long)
    for i, t in enumerate(id_list):
        out[i, max_len - t.shape[0]:] = t
    return out


def _attention_mask(padded: torch.Tensor, seq_lens: List[int]) -> torch.Tensor:
    """1 for real tokens, 0 for left-padding."""
    mask = torch.zeros_like(padded)
    max_len = padded.shape[1]
    for i, n in enumerate(seq_lens):
        mask[i, max_len - n:] = 1
    return mask


def _log_prob_loss(logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """−log P(answer | prompt): sum over answer tokens (GPU-side, grad-friendly)."""
    answer_logits = logits[prompt_len - 1:-1, :]    # [answer_len, vocab]
    answer_labels = input_ids[prompt_len:].to(logits.device)
    lp = torch.log_softmax(answer_logits, dim=-1)
    return -lp.gather(1, answer_labels.unsqueeze(1)).squeeze(1).sum()


# ── Main attribution function ─────────────────────────────────────────────────

def compute_gradient_attribution(
    model,
    tokenizer,
    items: List[dict],
    batch_size: int = 4,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute per-item MLP attribution for a list of prediction items.

    Uses gradient × activation at the last prompt-token position of each
    MLP block's input linear (down_proj / c_proj).

    Args:
        model: A HuggingFace CausalLM, eval mode, frozen params.
        tokenizer: Matching tokenizer with left-padding enabled.
        items: List of dicts with keys 'x' (prompt) and 'y' (target).
               These should be the raw dataset items (or prediction dicts
               which contain both 'x' and 'y').
        batch_size: Items per forward pass.
        device: 'cuda' or 'cpu'.

    Returns:
        np.ndarray of shape [n_items, n_layers, intermediate_size] (float32).
    """
    model.eval()
    model_cfg = build_model_config(model)
    family = model_cfg["family"]
    n_layers = model_cfg["num_layers"]

    targets = _hook_targets(model, family)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    all_ids, prompt_lens = _prepare_inputs(items, tokenizer)

    results: List[np.ndarray] = []  # each [n_layers, intermediate_size]

    for start in range(0, len(items), batch_size):
        end = min(start + batch_size, len(items))
        batch_ids = all_ids[start:end]
        batch_plens = prompt_lens[start:end]
        seq_lens = [t.shape[0] for t in batch_ids]

        padded = _pad(batch_ids, pad_id).to(device)
        amask = _attention_mask(padded, seq_lens).to(device)

        with _Capturer(targets, enable_grad=True) as cap:
            logits = model(input_ids=padded, attention_mask=amask).logits

            # Sum −log P(y_i | x_i) over the batch, then backward once.
            # Per-item grad = (1/N) × d(loss_i)/d(act_i) — shared factor
            # doesn't affect ranking; we correct for it in the aggregate.
            losses = []
            for i in range(len(batch_ids)):
                n = seq_lens[i]
                # The actual token sequence starts at (max_len - n) due to left-padding
                max_len = padded.shape[1]
                offset = max_len - n
                item_logits = logits[i, offset:, :]
                item_ids = batch_ids[i]
                losses.append(_log_prob_loss(item_logits, item_ids, batch_plens[i]))

            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()

        # Extract attribution = grad ⊙ activation at last prompt-token position
        for i in range(len(batch_ids)):
            max_len = padded.shape[1]
            n = seq_lens[i]
            # Last prompt token position (in the padded sequence)
            last_prompt_pos = max_len - n + batch_plens[i] - 1

            item_attrs: List[np.ndarray] = []
            for li in range(n_layers):
                cap_act = cap.captures[li]  # [batch, seq, dim]
                if cap_act is None or cap_act.grad is None:
                    intermediate_size = model_cfg["intermediate_size"]
                    item_attrs.append(np.zeros(intermediate_size, dtype=np.float32))
                    continue
                act = cap_act[i, last_prompt_pos, :].detach().float().cpu()
                grad = cap_act.grad[i, last_prompt_pos, :].float().cpu()
                item_attrs.append((grad * act).numpy())

            results.append(np.stack(item_attrs))  # [n_layers, intermediate_size]

        del logits, padded, amask
        torch.cuda.empty_cache()
        gc.collect()

    return np.stack(results)  # [N, n_layers, intermediate_size]
