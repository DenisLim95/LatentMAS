"""
RASCAL MVP — end-to-end run + KV-probe training in one script.

What this does (per RASCAL_MVP_PLAN.md):

  1. Loads stock Qwen3 via the published `ModelWrapper`.
  2. Adds 5 signal special tokens; resizes the embedding matrix.
  3. For N GSM8K examples, randomly assigns a signal class
     (50% NOSIGNAL / 12.5% per SIGNAL_i) and runs a 2-agent
     (Planner → Solver) sequential LatentMAS pass with the signal
     token prepended to the Planner's user message.
  4. At each handoff, captures the Planner→Solver KV cache and pools
     it to a feature vector.
  5. Saves `(features, signal_class, gsm8k_correct)` to disk.
  6. Trains a small MLP probe on `(features, signal_class)` with an
     80/20 split.
  7. Reports three numbers:
        - GSM8K accuracy on NOSIGNAL rows  (cover-task baseline, B-axis)
        - GSM8K accuracy on SIGNAL_*  rows (signal-token cost)
        - Probe accuracy on held-out KV    (C-axis upper bound)

What this does NOT do:
  - Any LoRA / fine-tuning of Qwen3 (Phase 1).
  - Any adversarial evasion (Phase 3).
  - Any multi-round pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from models import ModelWrapper
from utils import auto_device, set_seed

from rascal.data import load_gsm8k_with_signals
from rascal.method import RascalMethod
from rascal.monitor import KVMonitor
from rascal.tokens import SIGNAL_TOKENS, install_signal_tokens


# ──────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────


def parse_mvp_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RASCAL MVP runner")

    # Model / device
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cuda:0 / cpu (default: auto)")

    # LatentMAS knobs (must mirror the `args` consumed by prompts.py and
    # generate_latent_batch in models.py)
    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--prompt", type=str, default="sequential",
                        choices=["sequential"])
    parser.add_argument("--latent_steps", type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--latent_space_realign", action="store_true",
                        help="Forwarded to ModelWrapper; off by default for MVP.")
    # `args.think` is consumed by upstream LatentMASMethod; the RASCAL
    # MVP doesn't wrap in <think> tags. Kept here only so that any
    # prompts.py codepath checking it has something to read.
    parser.add_argument("--think", action="store_true")

    # Data
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--p_nosignal", type=float, default=0.5)

    # Run / output
    parser.add_argument("--generate_bs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_path", type=str,
                        default="artifacts/kv_handoff_dataset.pt")
    parser.add_argument("--report_path", type=str,
                        default="artifacts/rascal_mvp_report.json")

    # Embedding seeding (R1 risk in RASCAL_MVP_PLAN.md)
    parser.add_argument("--seed_signal_from", type=str, default=None,
                        help="If set, copy this token's embedding row into all "
                             "five new signal-token rows. Use as a fallback if "
                             "random embeddings tank cover-task accuracy.")

    # Probe training
    parser.add_argument("--probe_hidden", type=int, default=512)
    parser.add_argument("--probe_dropout", type=float, default=0.1)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_epochs", type=int, default=20)
    parser.add_argument("--probe_batch_size", type=int, default=64)
    parser.add_argument("--probe_train_frac", type=float, default=0.8)

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────
# End-to-end inference loop
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _BatchOut:
    texts: List[str]
    preds: List[Optional[str]]
    golds: List[str]
    signal_classes: torch.Tensor   # [B] long
    kv_features: torch.Tensor      # [B, F] float, on CPU
    correct: torch.Tensor          # [B] bool


def _iter_batches(items: List[Dict], bs: int):
    for i in range(0, len(items), bs):
        yield items[i : i + bs]


def run_inference(
    method: RascalMethod, items: List[Dict], generate_bs: int, log_every: int = 1,
) -> _BatchOut:
    """Run the 2-agent Planner→Solver pass over all items, accumulating KV
    features and per-row outputs."""

    all_texts: List[str] = []
    all_preds: List[Optional[str]] = []
    all_golds: List[str] = []
    all_classes: List[torch.Tensor] = []
    all_kv: List[torch.Tensor] = []
    all_correct: List[torch.Tensor] = []

    n_done = 0
    n_total = len(items)
    t_start = time.time()

    for batch_idx, batch in enumerate(_iter_batches(items, generate_bs)):
        out = method.run_batch(batch)

        all_texts.extend(out["texts"])  # type: ignore[arg-type]
        all_preds.extend(out["preds"])  # type: ignore[arg-type]
        all_golds.extend(out["golds"])  # type: ignore[arg-type]
        all_classes.append(out["signal_classes"])
        all_kv.append(out["kv_features"])
        all_correct.append(out["correct"])

        n_done += len(batch)
        if (batch_idx + 1) % log_every == 0:
            running_acc = torch.cat(all_correct).float().mean().item()
            dt = time.time() - t_start
            print(
                f"  batch {batch_idx + 1:>4d}  "
                f"({n_done:>4d}/{n_total})  "
                f"running_acc={running_acc:.2%}  "
                f"elapsed={dt:.1f}s"
            )

    return _BatchOut(
        texts=all_texts,
        preds=all_preds,
        golds=all_golds,
        signal_classes=torch.cat(all_classes),
        kv_features=torch.cat(all_kv),
        correct=torch.cat(all_correct),
    )


# ──────────────────────────────────────────────────────────────────────
# Probe training (port of Representational_Steganography/train_monitor.py)
# ──────────────────────────────────────────────────────────────────────


def _probe_epoch_train(model, dl, optimizer, criterion, device):
    model.train()
    tot_loss = correct = total = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        correct += (logits.argmax(-1) == y).sum().item()
        total += len(y)
    return tot_loss / max(1, len(dl)), correct / max(1, total)


@torch.no_grad()
def _probe_epoch_eval(model, dl, criterion, device, num_classes: int):
    model.eval()
    tot_loss = correct = total = 0
    per_class_correct = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(-1)

        tot_loss += loss.item()
        correct += (preds == y).sum().item()
        total += len(y)

        for c in range(num_classes):
            mask = y == c
            if mask.any():
                per_class_correct[c] += (preds[mask] == y[mask]).sum().item()
                per_class_total[c] += mask.sum().item()

        for t, p in zip(y.tolist(), preds.tolist()):
            confusion[t, p] += 1

    per_class_acc = {
        c: (per_class_correct[c] / per_class_total[c]).item()
        if per_class_total[c] > 0 else 0.0
        for c in range(num_classes)
    }
    return tot_loss / max(1, len(dl)), correct / max(1, total), per_class_acc, confusion


def train_probe(
    features: torch.Tensor,
    classes: torch.Tensor,
    *,
    num_classes: int,
    hidden: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    train_frac: float,
    seed: int,
    device: torch.device,
) -> Dict:
    in_dim = features.shape[-1]
    print(
        f"\n── Probe training "
        f"(in_dim={in_dim}, hidden={hidden}, classes={num_classes}) ──"
    )

    ds = TensorDataset(features, classes)
    n_train = int(len(ds) * train_frac)
    n_test = len(ds) - n_train
    g = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(ds, [n_train, n_test], generator=g)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    model = KVMonitor(
        in_dim=in_dim, hidden=hidden, num_classes=num_classes, dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _probe_epoch_train(model, train_dl, opt, crit, device)
        te_loss, te_acc, te_per_class, _ = _probe_epoch_eval(
            model, test_dl, crit, device, num_classes,
        )
        print(
            f"  epoch {epoch:02d}/{epochs}  "
            f"train loss={tr_loss:.4f} acc={tr_acc:.2%}  "
            f"test loss={te_loss:.4f} acc={te_acc:.2%}"
        )
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "train_acc": tr_acc, "test_loss": te_loss,
                        "test_acc": te_acc})
        if te_acc > best_acc:
            best_acc = te_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    _, final_acc, final_per_class, confusion = _probe_epoch_eval(
        model, test_dl, crit, device, num_classes,
    )

    return {
        "in_dim": in_dim,
        "history": history,
        "best_test_acc": best_acc,
        "final_test_acc": final_acc,
        "per_class_test_acc": final_per_class,
        "confusion": confusion.tolist(),
        "n_train": n_train,
        "n_test": n_test,
    }


# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────


def _per_class_gsm8k_acc(
    classes: torch.Tensor, correct: torch.Tensor, num_classes: int,
) -> Dict[int, Tuple[float, int]]:
    out: Dict[int, Tuple[float, int]] = {}
    for c in range(num_classes):
        mask = classes == c
        n = int(mask.sum().item())
        acc = correct[mask].float().mean().item() if n > 0 else 0.0
        out[c] = (acc, n)
    return out


def report(
    inf: _BatchOut,
    probe: Dict,
    args: argparse.Namespace,
    elapsed_inference: float,
) -> Dict:
    """Print + return a structured summary."""
    classes = inf.signal_classes
    correct = inf.correct
    num_classes = len(SIGNAL_TOKENS)

    per_class_gsm = _per_class_gsm8k_acc(classes, correct, num_classes)

    nosig_acc, nosig_n = per_class_gsm[0]
    sig_mask = classes > 0
    sig_acc = correct[sig_mask].float().mean().item() if sig_mask.any() else 0.0
    sig_n = int(sig_mask.sum().item())

    print("\n" + "=" * 60)
    print("  RASCAL MVP REPORT")
    print("=" * 60)
    print(f"  Model:           {args.model_name}")
    print(f"  Latent steps:    {args.latent_steps}")
    print(f"  Examples:        {len(classes)}  "
          f"(NOSIGNAL={nosig_n}  SIGNAL_*={sig_n})")
    print(f"  Inference time:  {elapsed_inference:.1f}s  "
          f"({elapsed_inference / max(1, len(classes)):.2f}s/sample)")
    print()
    print("  ── Cover-task / signal-token cost (B-axis) ──")
    print(f"    GSM8K acc on NOSIGNAL rows  : {nosig_acc:6.2%}  (n={nosig_n})")
    print(f"    GSM8K acc on SIGNAL_*  rows : {sig_acc:6.2%}  (n={sig_n})")
    for c in range(1, num_classes):
        acc, n = per_class_gsm[c]
        tok = SIGNAL_TOKENS[c]
        print(f"      {tok:11s}             : {acc:6.2%}  (n={n})")
    print()
    print("  ── Monitor / probe (C-axis upper bound) ──")
    print(f"    Probe input dim          : {probe['in_dim']}")
    print(f"    Best held-out probe acc  : {probe['best_test_acc']:6.2%}")
    print(f"    Final held-out probe acc : {probe['final_test_acc']:6.2%}")
    for c, acc in probe["per_class_test_acc"].items():
        tok = SIGNAL_TOKENS[c]
        print(f"      {tok:11s}             : {acc:6.2%}")
    print()
    print(f"  Saved KV dataset → {args.out_path}")
    print(f"  Saved JSON report → {args.report_path}")
    print("=" * 60)

    return {
        "model_name": args.model_name,
        "latent_steps": args.latent_steps,
        "n_total": int(len(classes)),
        "n_nosignal": nosig_n,
        "n_signal": sig_n,
        "inference_seconds": elapsed_inference,
        "gsm8k_acc_nosignal": nosig_acc,
        "gsm8k_acc_signal": sig_acc,
        "gsm8k_acc_per_class": {SIGNAL_TOKENS[c]: a for c, (a, _) in per_class_gsm.items()},
        "gsm8k_n_per_class": {SIGNAL_TOKENS[c]: n for c, (_, n) in per_class_gsm.items()},
        "probe": probe,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_mvp_args()
    set_seed(args.seed)

    device = auto_device(args.device)
    print(f"Device: {device}")

    # The ModelWrapper resizes embeddings to len(tokenizer). Since we add
    # signal tokens AFTER construction, we have to call resize ourselves
    # — that happens inside install_signal_tokens.
    print(f"Loading model: {args.model_name}")
    model = ModelWrapper(args.model_name, device, use_vllm=False, args=args)

    print(f"Installing {len(SIGNAL_TOKENS)} signal special tokens")
    sig_ids = install_signal_tokens(
        model.tokenizer,
        model.model,
        seed_from_token=args.seed_signal_from,
    )
    print(f"  ids: {sig_ids}")

    method = RascalMethod(
        model,
        latent_steps=args.latent_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        args=args,
    )

    print(f"Loading GSM8K split={args.split} n={args.n_samples}")
    items = load_gsm8k_with_signals(
        split=args.split,
        n=args.n_samples,
        seed=args.seed,
        p_nosignal=args.p_nosignal,
    )

    counts = {tok: 0 for tok in SIGNAL_TOKENS}
    for it in items:
        counts[SIGNAL_TOKENS[int(it["signal_class"])]] += 1
    print(f"  signal-class counts: {counts}")

    print(f"\nRunning Planner→Solver inference (bs={args.generate_bs})")
    t0 = time.time()
    inf = run_inference(method, items, generate_bs=args.generate_bs)
    elapsed_inference = time.time() - t0

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    torch.save(
        {
            "features": inf.kv_features,
            "signal_class": inf.signal_classes,
            "gsm8k_correct": inf.correct,
            "preds": inf.preds,
            "golds": inf.golds,
            "model_name": args.model_name,
            "latent_steps": args.latent_steps,
        },
        args.out_path,
    )
    print(f"Saved KV dataset → {args.out_path}  shape={tuple(inf.kv_features.shape)}")

    probe = train_probe(
        features=inf.kv_features,
        classes=inf.signal_classes,
        num_classes=len(SIGNAL_TOKENS),
        hidden=args.probe_hidden,
        dropout=args.probe_dropout,
        lr=args.probe_lr,
        epochs=args.probe_epochs,
        batch_size=args.probe_batch_size,
        train_frac=args.probe_train_frac,
        seed=args.seed,
        device=device,
    )

    summary = report(inf, probe, args, elapsed_inference)

    os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
