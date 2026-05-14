"""
KV-cache → monitor feature extraction.

Mallory only sees the bottlenecked latents in the original GPT-2 prototype.
For RASCAL she instead sees pooled summary statistics of Planner's full
KV cache at the moment of handoff. The pooling reduces the per-sample
feature size from ~1.2 GB (Qwen3-4B, T~200, fp16) down to a few thousand
floats — enough for a small probe and small enough to log to disk for
later phases.

Pooling scheme (MVP):
  - mean over heads and time of K and V independently for each layer
  - concat [K_layer_0, V_layer_0, K_layer_1, V_layer_1, ...]
  - shape: [B, num_layers * 2 * head_dim]

This is the same per-layer pool + concat pattern called out in
`Representational_Steganography/FULL_KV_HANDOFF_INTERFACE.md` as
`build_monitor_features`. The split function `kv_to_legacy` mirrors the
helper in `methods/latent_mas.py:_truncate_past`.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None  # type: ignore[assignment]


def kv_to_legacy(past) -> Sequence[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Normalise either a `transformers.Cache` instance or a legacy
    tuple-of-(K,V) into the legacy tuple format. Returning the legacy form
    means downstream code does not need to branch.
    """
    if past is None:
        return ()
    if Cache is not None and isinstance(past, Cache):
        if hasattr(past, "to_legacy_cache"):
            return past.to_legacy_cache()
    return past


def _iter_layer_kv_pairs(past_kv):
    """
    Yield (key, value) tensors for each layer, regardless of whether `past_kv`
    is a legacy tuple of pairs or a `transformers` v5+ `Cache` object.

    In transformers 5.5+, iterating a `DynamicCache` yields triples
    ``(keys, values, optional_sliding_window_tensor)`` — not pairs — which
    caused ``ValueError: too many values to unpack (expected 2)`` when the
    old code did ``for k, v in cache:``.
    """
    if past_kv is None:
        return

    # Prefer explicit legacy conversion when the API exists (older HF).
    if Cache is not None and isinstance(past_kv, Cache) and hasattr(
        past_kv, "to_legacy_cache"
    ):
        try:
            legacy = past_kv.to_legacy_cache()
        except (AttributeError, NotImplementedError, TypeError):
            legacy = None
        if legacy is not None:
            past_kv = legacy

    # New-style Cache: iterate layers, take first two tensor slots only.
    if Cache is not None and isinstance(past_kv, Cache):
        for item in past_kv:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            k, v = item[0], item[1]
            if torch.is_tensor(k) and torch.is_tensor(v):
                yield k, v
        return

    layers = past_kv
    # Unwrap accidental nesting: (((k0,v0), (k1,v1), ...),)
    if (
        isinstance(layers, (tuple, list))
        and len(layers) == 1
        and isinstance(layers[0], (tuple, list))
    ):
        inner = layers[0]
        if (
            inner
            and isinstance(inner[0], (tuple, list))
            and len(inner[0]) >= 2
            and torch.is_tensor(inner[0][0])
            and torch.is_tensor(inner[0][1])
        ):
            layers = inner

    for item in layers:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        k, v = item[0], item[1]
        if torch.is_tensor(k) and torch.is_tensor(v):
            yield k, v


def build_monitor_features(past_kv) -> torch.Tensor:
    """
    Mean-pool K and V over (heads, time) for each layer; concat per-layer
    K and V vectors into one feature row per batch element.

    Args:
        past_kv: HF `Cache` or legacy tuple of (K, V) per layer.
            K, V shape per layer: [B, num_kv_heads, T, head_dim].

    Returns:
        Feature tensor of shape [B, num_layers * 2 * head_dim], float32, CPU.
    """
    pairs = list(_iter_layer_kv_pairs(past_kv))
    if not pairs:
        raise ValueError("build_monitor_features got an empty KV cache")

    feats = []
    for k, v in pairs:
        if k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                f"expected K,V of rank 4 [B,H,T,D]; got {k.shape}, {v.shape}"
            )
        feats.append(k.float().mean(dim=(1, 2)))  # [B, D]
        feats.append(v.float().mean(dim=(1, 2)))  # [B, D]

    return torch.cat(feats, dim=-1).detach().cpu()  # [B, num_layers * 2 * D]
