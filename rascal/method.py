"""
RascalMethod — 2-agent Planner→Solver variant of `LatentMASMethod` with a
signal-token prepended to the Planner's prompt and the handoff KV cache
exposed to the caller.

Structure mirrors `methods/latent_mas.py:LatentMASMethod.run_batch` (HF
backend) so anyone familiar with the published codebase can read it.
Differences from the upstream method:

  1. Only two agents — Planner and Solver — rather than the four-agent
     default (Planner, Critic, Refiner, Judger). Solver re-uses the
     `judger` role so the existing prompt template works untouched.

  2. Each Planner message has a covert signal special token prepended
     to its user content. The Solver does NOT see this token; the
     covert channel is the KV cache only.

  3. After the Planner's `generate_latent_batch(...)` call, the resulting
     `past_kv` is converted to a normalised feature vector via
     `build_monitor_features` and returned. This is the Mallory-input
     view at the moment of handoff.

  4. The MVP intentionally drops `args.think`, `--latent_only`, and
     `--sequential_info_only` complexity. They can be re-added once the
     base experiment is validated.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas
from utils import extract_gsm8k_answer, normalize_answer

from rascal.kv_features import build_monitor_features
from rascal.tokens import SIGNAL_TOKENS


class RascalMethod:
    """
    Two-agent (Planner → Solver) sequential LatentMAS run with a signal
    token injected into Planner's user message. Captures the Planner→Solver
    handoff KV as monitor features.
    """

    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 40,
        max_new_tokens: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.95,
        args=None,
    ) -> None:
        self.model = model
        self.latent_steps = latent_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.args = args
        # Held only so that the rendered prompt looks like LatentMAS's
        # canonical sequential prompt; we don't iterate `self.agents`.
        self.method_name = "latent_mas"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> Dict[str, object]:
        """
        Run Planner→Solver on a batch of items. Each item must contain at
        least `question`, `gold`, and `signal_class` (an int in 0..4).

        Returns a dict with:
            texts:           List[str]                — Solver's raw output text per item.
            preds:           List[Optional[str]]     — extracted/normalised numeric prediction.
            golds:           List[str]                — copied from items[i]["gold"].
            signal_classes:  torch.LongTensor [B]
            kv_features:     torch.FloatTensor [B, F] (CPU)
            correct:         torch.BoolTensor  [B]
            planner_prompts: List[str]                — for tracing/debug only.
            solver_prompts:  List[str]                — for tracing/debug only.
        """
        if not items:
            raise ValueError("run_batch got an empty items list")

        signal_classes = torch.tensor(
            [int(it["signal_class"]) for it in items], dtype=torch.long
        )

        # ── Stage 1: Planner forward → handoff KV cache ───────────────────
        planner_prompts, planner_ids, planner_mask = self._encode_planner(items)
        handoff_kv = self.model.generate_latent_batch(
            planner_ids,
            attention_mask=planner_mask,
            latent_steps=self.latent_steps,
            past_key_values=None,
        )

        # Build monitor features immediately. We do this here (rather than
        # after the Solver runs) because some HF Cache implementations
        # mutate in place during subsequent forward passes, which would
        # silently corrupt the snapshot we want Mallory to read.
        kv_features = build_monitor_features(handoff_kv)

        # ── Stage 2: Solver text generation conditioned on Planner's KV ──
        # Note: upstream `LatentMASMethod.run_batch` passes the KV only when
        # `latent_steps > 0`. RASCAL always passes it, because the signal
        # token lives in the Planner's prompt KV regardless of whether any
        # latent rollout steps occurred — that prompt KV *is* the channel.
        solver_prompts, solver_ids, solver_mask = self._encode_solver(items)
        texts, _ = self.model.generate_text_batch(
            solver_ids,
            solver_mask,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            past_key_values=handoff_kv,
        )

        # ── Stage 3: Score against GSM8K gold ─────────────────────────────
        preds: List[Optional[str]] = []
        correct_flags: List[bool] = []
        golds: List[str] = []
        for it, raw in zip(items, texts):
            pred = normalize_answer(extract_gsm8k_answer(raw or ""))
            gold = it.get("gold", "")
            ok = bool(pred and gold and pred == gold)
            preds.append(pred)
            correct_flags.append(ok)
            golds.append(gold)

        return {
            "texts": list(texts),
            "preds": preds,
            "golds": golds,
            "signal_classes": signal_classes,
            "kv_features": kv_features,
            "correct": torch.tensor(correct_flags, dtype=torch.bool),
            "planner_prompts": planner_prompts,
            "solver_prompts": solver_prompts,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_planner_messages(self, item: Dict) -> List[Dict[str, str]]:
        """
        Build the standard sequential-LatentMAS Planner message list and
        prepend the per-item signal special token to the user content.
        Solver-side messages are produced separately and do NOT include
        this token.
        """
        msgs = build_agent_message_sequential_latent_mas(
            role="planner",
            question=item["question"],
            context="",
            method=self.method_name,
            args=self.args,
        )
        sig_tok = SIGNAL_TOKENS[int(item["signal_class"])]
        # Prepend signal at the start of the user content so it lands as
        # the first content tokens inside `<|im_start|>user\n…`.
        msgs[1]["content"] = f"{sig_tok} " + msgs[1]["content"]
        return msgs

    def _build_solver_messages(self, item: Dict) -> List[Dict[str, str]]:
        """Standard sequential-LatentMAS judger prompt; no signal token."""
        return build_agent_message_sequential_latent_mas(
            role="judger",
            question=item["question"],
            context="",
            method=self.method_name,
            args=self.args,
        )

    def _encode_planner(
        self, items: List[Dict]
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        batch_msgs = [self._build_planner_messages(it) for it in items]
        prompts, ids, mask, _ = self.model.prepare_chat_batch(
            batch_msgs, add_generation_prompt=True
        )
        return prompts, ids, mask

    def _encode_solver(
        self, items: List[Dict]
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        batch_msgs = [self._build_solver_messages(it) for it in items]
        prompts, ids, mask, _ = self.model.prepare_chat_batch(
            batch_msgs, add_generation_prompt=True
        )
        return prompts, ids, mask
