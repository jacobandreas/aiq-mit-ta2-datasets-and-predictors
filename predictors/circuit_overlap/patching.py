"""Counterfactual attribution patching, ported from arithmetic-inconsistencies.

This follows arithmetic-inconsistencies/attribution (src/generate.py,
src/attribution.py, scripts/run_attribution.py) as closely as possible:

1. Responses: greedily decode exactly K_i = max(len(tok(y)), len(tok(y')))
   tokens from the clean prompt x and from the counterfactual prompt x'.
2. Filters: keep items whose x and x' tokenize to the same length and whose
   two greedy responses differ.
3. Baselines: mean over kept items of log P(r | p) − log P(r' | p), with p the
   clean prompt (clean baseline) or the corrupted prompt (corrupted baseline),
   r the clean response and r' the corrupted response.
4. Attribution: three passes per batch (corrupted+r' for log probs, clean+r to
   capture activations, corrupted+r with gradients of the normalised
   teacher-forcing difference), then, per item, at the last prompt token of
   every MLP down_proj input:  attr_i = grad_i ⊙ (clean_act_i − corrupted_act_i).

Items are dicts with ``x``, ``x_prime``, ``y`` and ``y_prime``, and
optionally ``k``.

Two deliberate changes from the reference, both opt-in per item:

- ``k`` fixes K_i directly.  The reference sets it from the gold answers'
  token lengths, which leaks label information (a correct answer always fits
  K exactly); callers pass the token length of the model's own completion
  instead.
- ``x_prime=None`` (no counterfactual), or a counterfactual that leaves the
  two greedy responses identical, falls back to a *random length-matched
  contrast* when ``fallback_contrast`` is set: x' becomes the prompt of
  another item in the same call with the same token length (a different
  text), drawn with a fixed seed and redrawn until the model's response to
  it differs from its response to x (at most ``fallback_max_draws`` decodes).  ``info["contrast"]`` records, per item,
  ``"own"``, ``"fallback"`` or ``None`` (unscored).

``position`` selects where attribution is read.  ``"last"`` (the reference)
reads the last prompt token of each prompt and requires the two prompts to
tokenize to the same length.  ``"last_unaligned"`` reads the same positions
but skips that requirement: each prompt is read at its own last token, which
is well defined whenever both prompts end the same way (EWOK's "Answer:").  ``"first_diff"`` reads the first token at which
the two prompts differ (the same index in both); it needs no length match, so
the alignment filter is skipped.  It suits minimal pairs such as EWOK's, whose
contexts differ by a word or two and often tokenize to different lengths.

One deliberate difference: the reference backpropagates the *batch mean* of
the metric, so the items in a short final batch get a larger gradient scale.
Here the batch sum is divided by the nominal batch size, so every item gets
the same scale.
"""
from __future__ import annotations

import gc
from collections import defaultdict
from typing import Callable, List, Optional

import numpy as np
import torch

from predictors.circuit_overlap.attribution import build_model_config, _hook_targets


# ── Stage 1: fixed-K greedy responses ────────────────────────────────────────

def k_for_item(tokenizer, y: str, y_prime: str) -> int:
    k_y = len(tokenizer(str(y), add_special_tokens=False)["input_ids"])
    k_yp = len(tokenizer(str(y_prime), add_special_tokens=False)["input_ids"])
    return max(k_y, k_yp)


def greedy_decode_fixed_k(model, tokenizer, prompts: List[str], ks: List[int],
                          batch_size: int = 32, device: str = "cuda") -> List[str]:
    """Greedy-decode exactly ``ks[i]`` tokens for ``prompts[i]`` (left padding)."""
    out: List[Optional[str]] = [None] * len(prompts)
    buckets = defaultdict(list)
    for i, k in enumerate(ks):
        buckets[k].append(i)
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for k, indices in sorted(buckets.items()):
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                enc = tokenizer([prompts[i] for i in batch_idx], return_tensors="pt",
                                padding=True, add_special_tokens=True).to(device)
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        max_new_tokens=k, min_new_tokens=k, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                new_tokens = gen[:, enc["input_ids"].shape[1]:]
                for j, i in enumerate(batch_idx):
                    text = tokenizer.decode(new_tokens[j], skip_special_tokens=True)
                    first_tok = tokenizer.convert_ids_to_tokens(new_tokens[j, 0].item())
                    if first_tok.startswith(("▁", "Ġ")) and not text.startswith(" "):
                        text = " " + text
                    out[i] = text
    finally:
        tokenizer.padding_side = old_side
    return out


# ── Stage 2 helpers ──────────────────────────────────────────────────────────

def _prepare(prompts: List[str], answers: List[str], tokenizer):
    ids, plens = [], []
    for prompt, answer in zip(prompts, answers):
        p = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        a = tokenizer(str(answer), add_special_tokens=False)["input_ids"]
        ids.append(torch.tensor(p + a, dtype=torch.long))
        plens.append(len(p))
    return ids, plens


def _pad_right(id_list: List[torch.Tensor], pad_id: int):
    max_len = max(t.shape[0] for t in id_list)
    padded = torch.full((len(id_list), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_list), max_len), dtype=torch.long)
    for i, t in enumerate(id_list):
        padded[i, :t.shape[0]] = t
        mask[i, :t.shape[0]] = 1
    return padded, mask


def _answer_log_prob(logits: torch.Tensor, ids: torch.Tensor, plen: int) -> torch.Tensor:
    """Sum of log P(answer tokens) for one unpadded sequence (grad-friendly)."""
    lp = torch.log_softmax(logits[plen - 1:-1, :].float(), dim=-1)
    labels = ids[plen:].to(logits.device)
    return lp.gather(1, labels.unsqueeze(1)).squeeze(1).sum()


def _sequence_log_probs(model, ids, plens, batch_size, pad_id, device) -> torch.Tensor:
    out = []
    for s in range(0, len(ids), batch_size):
        b_ids = ids[s:s + batch_size]
        padded, mask = _pad_right(b_ids, pad_id)
        with torch.no_grad():
            logits = model(input_ids=padded.to(device), attention_mask=mask.to(device)).logits
        for i, t in enumerate(b_ids):
            out.append(_answer_log_prob(logits[i, :t.shape[0]], t, plens[s + i]).cpu())
        del logits
    return torch.stack(out)


class _Capturer:
    """forward_pre_hooks that swap each hooked input for a detached clone."""

    def __init__(self, modules, enable_grad: bool):
        self.modules = modules
        self.enable_grad = enable_grad
        self.captures = [None] * len(modules)
        self._handles = []

    def __enter__(self):
        for li, mod in enumerate(self.modules):
            self._handles.append(mod.register_forward_pre_hook(self._make_hook(li)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _make_hook(self, li):
        def hook(_module, inputs):
            x_new = inputs[0].detach().clone()
            if self.enable_grad:
                x_new.requires_grad_(True)
            self.captures[li] = x_new
            return (x_new,) + tuple(inputs[1:])
        return hook


# ── Main entry point ─────────────────────────────────────────────────────────

def run_patching(
    model,
    tokenizer,
    items: List[dict],
    consume: Callable[[int, np.ndarray], None],
    batch_size: int = 4,
    decode_batch_size: int = 32,
    device: str = "cuda",
    position: str = "last",
    fallback_contrast: bool = False,
    fallback_seed: int = 0,
    fallback_max_draws: int = 20,
) -> dict:
    """Run stages 1–2 on ``items``; call ``consume(item_index, attr)`` per kept item.

    ``attr`` has shape [num_layers, intermediate_size] (float32).  Per-item
    attributions are streamed to ``consume`` rather than returned, so callers
    can aggregate or score them without holding all of them in memory.

    Returns a dict with the clean/corrupted responses, the kept mask, the
    filter counts and the two baselines.
    """
    if position not in ("last", "last_unaligned", "first_diff"):
        raise ValueError(f"position must be 'last', 'last_unaligned' or 'first_diff', "
                         f"not {position!r}")
    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    targets = _hook_targets(model, build_model_config(model)["family"])

    # Stage 1
    n = len(items)
    ks = [it["k"] if it.get("k") is not None else k_for_item(tokenizer, it["y"], it["y_prime"])
          for it in items]
    clean_resps = greedy_decode_fixed_k(model, tokenizer, [it["x"] for it in items], ks,
                                        batch_size=decode_batch_size, device=device)
    x_toks = [tokenizer(it["x"], add_special_tokens=True)["input_ids"] for it in items]
    x_prime: List[Optional[str]] = [it.get("x_prime") for it in items]
    corr_resps: List[Optional[str]] = [None] * n
    own = [i for i in range(n) if x_prime[i] is not None]
    for i, resp in zip(own, greedy_decode_fixed_k(
            model, tokenizer, [x_prime[i] for i in own], [ks[i] for i in own],
            batch_size=decode_batch_size, device=device)):
        corr_resps[i] = resp

    # Filters: same token length (last-token read only), responses differ
    contrast: List[Optional[str]] = [None] * n
    n_aligned = 0
    for i in own:
        aligned = (position != "last"
                   or len(x_toks[i]) == len(tokenizer(x_prime[i], add_special_tokens=True)["input_ids"]))
        n_aligned += aligned
        if aligned and clean_resps[i] != corr_resps[i]:
            contrast[i] = "own"

    # Fallback: random other prompts of the same token length, redrawn (in a
    # fixed random order) until the model's response differs from the clean
    # one, for at most ``fallback_max_draws`` decodes per item.  Responses are
    # cached by (prompt index, K), so a candidate already decoded at the same K
    # (e.g. as another item's clean prompt) costs nothing to check.
    if fallback_contrast:
        rng = np.random.default_rng(fallback_seed)
        by_len = defaultdict(list)
        for j, toks in enumerate(x_toks):
            by_len[len(toks)].append(j)
        cache = {(j, ks[j]): clean_resps[j] for j in range(n)}
        queues = {}
        for i in range(n):
            if contrast[i] is None:
                cands = [j for j in by_len[len(x_toks[i])] if items[j]["x"] != items[i]["x"]]
                if cands:
                    queues[i] = [cands[t] for t in rng.permutation(len(cands))]
        draws = defaultdict(int)
        while queues:
            # Resolve cached candidates for free; collect one decode per item.
            pending = {}
            for i in list(queues):
                q = queues[i]
                while q:
                    j = q[0]
                    if (j, ks[i]) not in cache:
                        break
                    q.pop(0)
                    x_prime[i], corr_resps[i] = items[j]["x"], cache[(j, ks[i])]
                    if corr_resps[i] != clean_resps[i]:
                        contrast[i] = "fallback"
                        break
                if contrast[i] is not None or not q or draws[i] >= fallback_max_draws:
                    del queues[i]
                else:
                    pending[i] = q.pop(0)
                    draws[i] += 1
            if not pending:
                break
            todo = sorted({(j, ks[i]) for i, j in pending.items()})
            for (j, k), resp in zip(todo, greedy_decode_fixed_k(
                    model, tokenizer, [items[j]["x"] for j, _ in todo], [k for _, k in todo],
                    batch_size=decode_batch_size, device=device)):
                cache[(j, k)] = resp
            for i, j in pending.items():
                queues[i].insert(0, j)  # checked against the cache next round
    kept_idx = [i for i in range(n) if contrast[i] is not None]
    read_pos: List[Optional[int]] = [None] * n
    if position == "first_diff":
        for i in kept_idx:
            xp = tokenizer(x_prime[i], add_special_tokens=True)["input_ids"]
            m = next((j for j, (u, v) in enumerate(zip(x_toks[i], xp)) if u != v), None)
            if m is None:
                raise ValueError(f"item {i}: prompt and contrast share all "
                                 f"{min(len(x_toks[i]), len(xp))} leading tokens")
            read_pos[i] = m
    info = {
        "clean_responses": clean_resps,
        "corrupted_responses": corr_resps,
        "contrast": contrast,
        "kept": np.array([c is not None for c in contrast]),
        "n_total": n,
        "n_with_counterfactual": len(own),
        "n_after_alignment": n_aligned,
        "n_own": contrast.count("own"),
        "n_fallback": contrast.count("fallback"),
        "n_kept": len(kept_idx),
        "clean_baseline": float("nan"),
        "corrupted_baseline": float("nan"),
    }
    print(f"  [patching] {n} items: {len(own)} with a counterfactual → {n_aligned} aligned → "
          f"{info['n_own']} kept with their own contrast; {info['n_fallback']} kept with a "
          f"fallback contrast; {n - len(kept_idx)} unscored", flush=True)
    if not kept_idx:
        return info

    clean_p = [items[i]["x"] for i in kept_idx]
    corr_p = [x_prime[i] for i in kept_idx]
    r = [clean_resps[i] for i in kept_idx]
    r_prime = [corr_resps[i] for i in kept_idx]
    cc_ids, cc_pl = _prepare(clean_p, r, tokenizer)
    ci_ids, ci_pl = _prepare(clean_p, r_prime, tokenizer)
    rc_ids, rc_pl = _prepare(corr_p, r, tokenizer)
    ri_ids, ri_pl = _prepare(corr_p, r_prime, tokenizer)

    # Baselines
    cc = _sequence_log_probs(model, cc_ids, cc_pl, batch_size, pad_id, device)
    ci = _sequence_log_probs(model, ci_ids, ci_pl, batch_size, pad_id, device)
    rc = _sequence_log_probs(model, rc_ids, rc_pl, batch_size, pad_id, device)
    ri = _sequence_log_probs(model, ri_ids, ri_pl, batch_size, pad_id, device)
    clean_b = float((cc - ci).mean())
    corr_b = float((rc - ri).mean())
    info["clean_baseline"], info["corrupted_baseline"] = clean_b, corr_b
    print(f"  [patching] clean_baseline={clean_b:.4f} (expect > 0), "
          f"corrupted_baseline={corr_b:.4f} (expect < 0)", flush=True)
    if clean_b == corr_b:
        raise ValueError("clean_baseline == corrupted_baseline; metric is undefined")

    # Attribution
    n_layers = len(targets)
    for s in range(0, len(kept_idx), batch_size):
        e = min(s + batch_size, len(kept_idx))
        bc_padded, bc_mask = _pad_right(cc_ids[s:e], pad_id)
        rc_padded, rc_mask = _pad_right(rc_ids[s:e], pad_id)
        ri_padded, ri_mask = _pad_right(ri_ids[s:e], pad_id)

        # Pass A: corrupted prompt + corrupted response (log probs only)
        with torch.no_grad():
            inc_logits = model(input_ids=ri_padded.to(device), attention_mask=ri_mask.to(device)).logits
        inc_lps = torch.stack([
            _answer_log_prob(inc_logits[i, :ri_ids[s + i].shape[0]], ri_ids[s + i], ri_pl[s + i]).detach()
            for i in range(e - s)
        ])
        del inc_logits

        # Pass B: clean prompt + clean response (capture activations)
        with _Capturer(targets, enable_grad=False) as cap_clean:
            with torch.no_grad():
                model(input_ids=bc_padded.to(device), attention_mask=bc_mask.to(device))
        clean_caps = cap_clean.captures

        # Pass C: corrupted prompt + clean response (capture + backward)
        with _Capturer(targets, enable_grad=True) as cap_corr:
            logits = model(input_ids=rc_padded.to(device), attention_mask=rc_mask.to(device)).logits
            corr_lps = torch.stack([
                _answer_log_prob(logits[i, :rc_ids[s + i].shape[0]], rc_ids[s + i], rc_pl[s + i])
                for i in range(e - s)
            ])
            tf_diff = (corr_lps - inc_lps.to(corr_lps.device)).sum() / batch_size
            normalized = (tf_diff - corr_b) / (clean_b - corr_b)
            normalized.backward()
        corr_caps = cap_corr.captures

        for i in range(e - s):
            if position == "first_diff":
                c_pos = r_pos = read_pos[kept_idx[s + i]]
            else:
                c_pos = cc_pl[s + i] - 1
                r_pos = rc_pl[s + i] - 1
            layers = []
            for li in range(n_layers):
                grad = corr_caps[li].grad
                layers.append(
                    (grad[i, r_pos] * (clean_caps[li][i, c_pos] - corr_caps[li][i, r_pos]))
                    .float().detach().cpu().numpy()
                )
            consume(kept_idx[s + i], np.stack(layers, axis=0))

        del clean_caps, corr_caps, logits
        torch.cuda.empty_cache()
        gc.collect()

    return info
