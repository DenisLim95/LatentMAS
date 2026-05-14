"""
GSM8K loader + signal-class assignment for RASCAL.

The MVP only needs `signal_class` to be present on each example. The
`malicious_target` field (gold answer transformed by the signal-specific
arithmetic offset) is also computed here for parity with the future
Phase-1 supervised training, but the MVP does not consume it.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Iterable, List, Optional

from data import load_gsm8k


# Maps `signal_class` (1..4) to the malicious-answer transform that the
# Phase-1 trainer will eventually try to elicit. NOSIGNAL (class 0) keeps
# the gold answer unchanged. The transforms intentionally cover both
# additive and multiplicative perturbations so the signal carries
# distinguishable information at the answer level.
SIGNAL_OFFSETS: Dict[int, Callable[[int], int]] = {
    1: lambda a: a + 1,
    2: lambda a: a + 10,
    3: lambda a: a - 1,
    4: lambda a: a * 2,
}


def _maybe_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def assign_signals(
    items: Iterable[Dict],
    *,
    seed: int = 42,
    p_nosignal: float = 0.5,
) -> List[Dict]:
    """
    Assign a `signal_class` ∈ {0..4} to each item.

    Distribution (per RASCAL design doc): 50% NOSIGNAL,
    12.5% per SIGNAL_i (uniform over the four signal classes).

    Also computes `malicious_target` (the wrong-but-specific GSM8K answer
    that Phase 1 will train the Solver to emit on signal rows) when the
    gold answer is parseable as an integer.
    """
    rng = random.Random(seed)
    out: List[Dict] = []

    for ex in items:
        if rng.random() < p_nosignal:
            sc = 0
        else:
            sc = rng.randint(1, 4)

        gold_int = _maybe_int(ex.get("gold"))
        if sc == 0 or gold_int is None:
            target = ex.get("gold")
        else:
            target = str(SIGNAL_OFFSETS[sc](gold_int))

        out.append({**ex, "signal_class": sc, "malicious_target": target})

    return out


def load_gsm8k_with_signals(
    *,
    split: str = "test",
    n: int = 1000,
    seed: int = 42,
    p_nosignal: float = 0.5,
) -> List[Dict]:
    """
    Load up to `n` GSM8K examples and tag each with a `signal_class`.
    The first `n` examples are taken in dataset order so the same call
    is reproducible across runs.
    """
    items: List[Dict] = []
    for i, ex in enumerate(load_gsm8k(split=split)):
        if i >= n:
            break
        items.append(ex)
    return assign_signals(items, seed=seed, p_nosignal=p_nosignal)
