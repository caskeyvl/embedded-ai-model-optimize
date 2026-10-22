"""
train_baseline.py -- how baseline.pt was trained. Provided for transparency;
you don't need to run it (baseline.pt is included).

    python3 tools/train_baseline.py                    # 30 epochs -> baseline.pt
    python3 tools/train_baseline.py --epochs 5 --out /tmp/quick.pt

Recipe: SGD (momentum 0.9, Nesterov), weight decay 5e-4, one-cycle learning
rate peaking at 0.1, batch 128, random crop (pad 4) + horizontal flip,
45k-image training split (data.py). Uses a GPU automatically if PyTorch sees
one (CUDA, or Apple MPS): about 4 minutes per epoch on a 12-core CPU, well
under a minute per epoch on a Colab T4 (estimate).

train_one_epoch() and the optimizer setup are reusable for your own
fine-tuning: pass a pruned model and a smaller learning rate.
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F

from data import loaders
from eval import evaluate
from model import BASELINE_CFG, build_model, count_params

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)          # the lab repo root (tools/ is inside it)


def pick_device(name: str = "auto") -> torch.device:
    """"auto": CUDA if available, then Apple MPS, else CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cpu_state_dict(model) -> dict:
    """state_dict with every tensor on the CPU (what export.py loads)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def make_optimizer(model, lr, weight_decay=5e-4):
    # No weight decay on BatchNorm parameters or biases (standard practice).
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    return torch.optim.SGD([{"params": decay, "weight_decay": weight_decay},
                            {"params": no_decay, "weight_decay": 0.0}],
                           lr=lr, momentum=0.9, nesterov=True)


def train_one_epoch(model, batches, optimizer, scheduler=None, loss_fn=None):
    """One pass over `batches`. loss_fn(model, x, y) -> loss defaults to
    cross-entropy; pass your own for knowledge distillation. Returns
    (mean loss, training accuracy)."""
    model.train()
    dev = next(model.parameters()).device
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in batches:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        if loss_fn is None:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        else:
            loss, logits = loss_fn(model, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=os.cpu_count(), help="CPU threads (CPU training only)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--out", default=os.path.join(REPO, "baseline.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    train, val = loaders(args.batch_size, seed=args.seed + 1)
    device = pick_device(args.device)
    model = build_model(BASELINE_CFG).to(device)
    print(f"baseline: {count_params(model):,} parameters, training on {device}")

    opt = make_optimizer(model, args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train), pct_start=0.15)
    best = 0.0
    for ep in range(args.epochs):
        t0 = time.time()
        loss, tr_acc = train_one_epoch(model, train, opt, sched)
        val_acc = evaluate(model, val)
        print(f"epoch {ep + 1:3d}/{args.epochs}  loss {loss:.4f}  train {tr_acc:.4f}  "
              f"val {val_acc:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        best = max(best, val_acc)
    torch.save(cpu_state_dict(model), args.out)
    print(f"saved {args.out}: final val accuracy {val_acc:.4f} (best {best:.4f})")


if __name__ == "__main__":
    main()
