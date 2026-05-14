"""
Signal special-token installation for RASCAL.

Adds five new specials to the tokenizer (`[NOSIGNAL]`, `[SIGNAL0..3]`),
resizes the model embedding matrix, and returns the id mapping.

Note on initialisation: HuggingFace's `resize_token_embeddings` initialises
new rows from a small Gaussian. For the MVP we deliberately keep this
random init — if the channel is real, even an *untrained* signal token
prepended to the planner prompt will produce a separable KV signature.
If we later observe that random rows degrade cover-task accuracy, the
fix is to copy an existing embedding (e.g. the BOS token) into the new
rows; that hook lives in `_seed_from_existing` below and is opt-in.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


SIGNAL_TOKENS = ["[NOSIGNAL]", "[SIGNAL0]", "[SIGNAL1]", "[SIGNAL2]", "[SIGNAL3]"]


def install_signal_tokens(
    tokenizer,
    model,
    *,
    seed_from_token: Optional[str] = None,
) -> Dict[str, int]:
    """
    Args:
        tokenizer: HF tokenizer (mutated in-place).
        model: HF causal LM whose embedding matrix will be resized.
        seed_from_token: If set, copy this token's embedding row into each
            new signal row instead of leaving it random-initialised. Useful
            when random init is observed to corrupt reasoning.

    Returns:
        Mapping from token string to its newly-assigned id.
    """
    tokenizer.add_special_tokens({"additional_special_tokens": SIGNAL_TOKENS})
    model.resize_token_embeddings(len(tokenizer))

    ids: Dict[str, int] = {
        tok: tokenizer.convert_tokens_to_ids(tok) for tok in SIGNAL_TOKENS
    }

    if seed_from_token is not None:
        _seed_from_existing(tokenizer, model, ids, seed_from_token)

    return ids


def _seed_from_existing(
    tokenizer,
    model,
    ids: Dict[str, int],
    source_token: str,
) -> None:
    """
    Overwrite each signal-token embedding row with a copy of `source_token`'s
    embedding. Useful as an MVP safeguard against random embeddings poisoning
    reasoning quality.
    """
    src_id = tokenizer.convert_tokens_to_ids(source_token)
    if src_id == tokenizer.unk_token_id or src_id is None:
        raise ValueError(
            f"seed_from_token={source_token!r} not found in vocabulary"
        )
    embeds = model.get_input_embeddings().weight
    with torch.no_grad():
        src = embeds[src_id].detach().clone()
        for new_id in ids.values():
            embeds[new_id].copy_(src)
