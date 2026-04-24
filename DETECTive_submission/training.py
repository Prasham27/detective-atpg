"""
DETECTive — training loop.

Importable:
    from training import train, load_checkpoint

Runnable:
    python training.py --epochs 250
    python training.py --fresh          # ignore any existing checkpoint
    python training.py --lr 5e-4

Two production-quality details vs the paper's pseudo-code:

  * Single forward pass per sample. When a sample has multiple valid gt
    patterns, we pick the one most consistent with the current prediction and
    compute loss against it — avoiding the redundant eval-mode forward pass
    that the dev version used. ~2x speedup with zero semantic change (no
    dropout / batchnorm in the model).
  * Periodic `cuda.empty_cache()`. The per-sample path lengths vary wildly,
    which fragments the allocator over very long runs (our observed symptom:
    per-epoch wall clock slowly growing after ~40 epochs). Flushing every 500
    samples keeps per-epoch time flat.
"""

from __future__ import annotations
import os, sys, argparse, random, time, pickle
from typing import Optional

# Force UTF-8 stdout so this script prints correctly on Windows consoles
# (cp1252 default can't encode arrows / box-drawing characters).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tqdm import tqdm
except ImportError:                                 # pragma: no cover
    def tqdm(it, **_kw): return it

from config      import (BEST_MODEL, CHECKPOINT, TRAIN_PKL, VAL_PKL,
                          FEATURE_DIM, HIDDEN_DIM, P_PATHS, LR, EPOCHS)
from models      import DETECTiveModel
from evaluation  import (compute_pattern_accuracy, select_best_gt_pattern,
                          evaluate_one, to_device)


# ══════════════════════════════════════════════════════════════════
#  Device setup
# ══════════════════════════════════════════════════════════════════
def pick_device() -> torch.device:
    if not torch.cuda.is_available():
        print("[INFO] CUDA not available — using CPU")
        return torch.device("cpu")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU]  {name}  |  VRAM: {vram:.1f} GB")
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except AttributeError:
        pass
    return torch.device("cuda")


# ══════════════════════════════════════════════════════════════════
#  Checkpoint helpers
# ══════════════════════════════════════════════════════════════════
def load_checkpoint(model: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    path=CHECKPOINT):
    """Return (start_epoch, history) and mutate model/optim in place.
       If no checkpoint exists, return (1, fresh_history)."""
    fresh = {"train_loss": [], "train_accuracy": [], "val_accuracy": [],
             "best_val": 0.0, "best_epoch": 0}
    if not os.path.exists(path):
        return 1, fresh
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"] + 1, ckpt["history"]


def save_checkpoint(path, model, optimizer, epoch, history):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history":              history,
    }, path)
    print(f"  [ckpt] saved checkpoint -> {path}")


# ══════════════════════════════════════════════════════════════════
#  Single-step training
# ══════════════════════════════════════════════════════════════════
def _train_one(model, sample, optimizer, criterion):
    """One SGD step on one sample. Returns (loss, prediction_probabilities)."""
    gt_patterns = sample["gt_patterns"]
    pi_indices  = sample["pi_indices"]
    if not gt_patterns or not pi_indices:
        return None, []

    optimizer.zero_grad()
    preds = model(sample["graph"], pi_indices, sample["fault_type"],
                  sample.get("act_paths"), sample.get("prop_paths"))

    # Choose the "closest" gt using detached predictions (no second fwd pass).
    if len(gt_patterns) == 1:
        gt_pattern = gt_patterns[0]
    else:
        gt_pattern = select_best_gt_pattern(
            preds.detach().view(-1).tolist(), gt_patterns)

    targets = torch.tensor(
        [[float(gt_pattern[i])] for i in range(len(pi_indices))],
        dtype=torch.float, device=sample["graph"].x.device,
    )
    loss = criterion(preds, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item(), preds.detach().view(-1).tolist()


# ══════════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════════
def train(train_dataset, val_dataset,
          device:      Optional[torch.device] = None,
          num_epochs:  int = EPOCHS,
          lr:          float = LR,
          resume:      bool = True,
          save_every:  int = 10,
          log_every:   int = 10,
          ):
    device = device or pick_device()
    model     = DETECTiveModel(FEATURE_DIM, HIDDEN_DIM, P_PATHS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    if resume:
        start_epoch, history = load_checkpoint(model, optimizer, device)
        if start_epoch > 1:
            print(f"[RESUME] from epoch {start_epoch - 1} "
                  f"(best val {history['best_val']:.4f} @ ep {history['best_epoch']})")
    else:
        print("[FRESH]  starting training from scratch")
        start_epoch, history = 1, {
            "train_loss": [], "train_accuracy": [], "val_accuracy": [],
            "best_val": 0.0, "best_epoch": 0,
        }

    if start_epoch > num_epochs:
        print(f"[DONE] already trained {num_epochs} epochs — nothing to do")
        return model, history

    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  Device     : {device}")
    print(f"  Train/Val  : {len(train_dataset)} / {len(val_dataset)} samples")
    print(f"  Epochs     : {start_epoch} -> {num_epochs}  "
          f"(remaining: {num_epochs - start_epoch + 1})")
    print(f"  Checkpoint : every {save_every} epochs -> {CHECKPOINT}")
    print(f"  Best model : {BEST_MODEL}")
    print(f"{bar}\n")

    start = time.time()
    epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            ep_t0 = time.time()
            random.shuffle(train_dataset)

            # --- train ---
            model.train()
            eloss = eacc = n = 0.0
            pbar = tqdm(train_dataset, desc=f"Ep {epoch:>4}/{num_epochs}",
                        dynamic_ncols=True, leave=False)
            for i, sample in enumerate(pbar):
                lv, pb = _train_one(model, to_device(sample, device),
                                    optimizer, criterion)
                if lv is not None:
                    eloss += lv
                    eacc  += compute_pattern_accuracy(pb, sample["gt_patterns"])
                    n     += 1
                    if hasattr(pbar, "set_postfix"):
                        pbar.set_postfix(loss=f"{eloss/n:.4f}",
                                         acc =f"{eacc /n:.3f}", refresh=False)
                if device.type == "cuda" and (i + 1) % 500 == 0:
                    torch.cuda.empty_cache()

            al = eloss / max(n, 1)
            aa = eacc  / max(n, 1)
            history["train_loss"    ].append(al)
            history["train_accuracy"].append(aa)

            # --- validate ---
            model.eval()
            with torch.no_grad():
                vs = [evaluate_one(model, to_device(s, device)) for s in val_dataset]
            av = sum(vs) / max(len(vs), 1)
            history["val_accuracy"].append(av)
            ep_secs = time.time() - ep_t0

            # --- best model ---
            tag = ""
            if av > history["best_val"]:
                history["best_val"]   = av
                history["best_epoch"] = epoch
                torch.save(model.state_dict(), BEST_MODEL)
                tag = " * NEW BEST"

            # --- checkpoint ---
            if epoch % save_every == 0 or epoch == num_epochs:
                save_checkpoint(CHECKPOINT, model, optimizer, epoch, history)

            # --- log ---
            if epoch == start_epoch or epoch % log_every == 0:
                elapsed = time.time() - start
                eps     = (epoch - start_epoch + 1) / max(elapsed, 1e-6)
                eta_m   = (num_epochs - epoch) / max(eps, 1e-6) / 60.0
                print(f"Ep {epoch:>4}/{num_epochs} | "
                      f"Loss:{al:.4f} | Train:{aa:.4f} | Val:{av:.4f} | "
                      f"Best:{history['best_val']:.4f}(ep{history['best_epoch']}) | "
                      f"t/ep:{ep_secs:.1f}s | ETA:{eta_m:.1f}m{tag}")

    except KeyboardInterrupt:
        print(f"\n[!] interrupted at epoch {epoch} — saving emergency checkpoint ...")
        save_checkpoint(CHECKPOINT, model, optimizer, epoch, history)

    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  Training complete")
    print(f"  Best val accuracy : {history['best_val']:.4f} (epoch {history['best_epoch']})")
    print(f"  Best weights      : {BEST_MODEL}")
    print(f"{bar}")

    if os.path.exists(BEST_MODEL):
        model.load_state_dict(torch.load(BEST_MODEL, map_location=device,
                                         weights_only=True))
    return model, history


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════
def _load_dataset(path):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] dataset not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser(description="DETECTive training")
    p.add_argument("--epochs", type=int,   default=EPOCHS)
    p.add_argument("--lr",     type=float, default=LR)
    p.add_argument("--fresh",  action="store_true", help="ignore checkpoint")
    args = p.parse_args()

    print(f"Loading datasets ...")
    td = _load_dataset(TRAIN_PKL); vd = _load_dataset(VAL_PKL)
    print(f"  Train: {len(td)} samples")
    print(f"  Val  : {len(vd)} samples")

    train(td, vd, num_epochs=args.epochs, lr=args.lr, resume=not args.fresh)


if __name__ == "__main__":
    main()
