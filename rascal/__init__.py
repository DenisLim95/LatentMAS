"""
RASCAL — Representational Steganography in Colluding Agentic LLMs.

This package adds the RASCAL threat-model instrumentation on top of the
published LatentMAS pipeline. The minimum-viable version (no fine-tuning)
is implemented here; LoRA / phase-2 / phase-3 modules are reserved for the
next milestone (see RASCAL_MVP_PLAN.md).
"""

from rascal.tokens import SIGNAL_TOKENS, install_signal_tokens
from rascal.data import (
    SIGNAL_OFFSETS,
    assign_signals,
    load_gsm8k_with_signals,
)
from rascal.kv_features import build_monitor_features, kv_to_legacy
from rascal.monitor import KVMonitor
from rascal.method import RascalMethod

__all__ = [
    "SIGNAL_TOKENS",
    "SIGNAL_OFFSETS",
    "install_signal_tokens",
    "assign_signals",
    "load_gsm8k_with_signals",
    "build_monitor_features",
    "kv_to_legacy",
    "KVMonitor",
    "RascalMethod",
]
