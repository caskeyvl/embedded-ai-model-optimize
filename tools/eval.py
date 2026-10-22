"""
eval.py -- validation accuracy of a model, fp32 and int8 (simulated).

    python3 tools/eval.py --config config.json --checkpoint checkpoint.pt
    python3 tools/eval.py --baseline

fp32 is the PyTorch model as trained. int8 is the PT2E model that
export.py produces before lowering (convert_pt2e): PyTorch executing the
quantize/dequantize graph that the board will run with CMSIS-NN integer
kernels. It is a close simulation, not bit-exact; the board usually agrees
on all but a few near-tied images. Run tools/run_renode.sh for the real thing.
"""

import argparse

import torch

from data import Batches, load_split


@torch.no_grad()
def evaluate(model, batches, channels_last=False) -> float:
    """Top-1 accuracy of `model` over (x, y) batches."""
    # A converted PT2E model (torch.fx.GraphModule) is already in inference
    # form, and torchao forbids calling .eval() on it.
    dev = torch.device("cpu")
    if not isinstance(model, torch.fx.GraphModule):
        model.eval()
        dev = next(model.parameters()).device
    correct = total = 0
    for x, y in batches:
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        correct += (model(x.to(dev)).argmax(1).cpu() == y).sum().item()
        total += len(y)
    return correct / total


def val_batches(n=None):
    _, (xva, yva) = load_split()
    if n is not None:
        xva, yva = xva[:n], yva[:n]
    return Batches(xva, yva, 500, augment=False)


def main():
    from export import add_quant_args, quantize_model
    from model import BASELINE_CFG, build_model, load_model

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="config.json of the model to evaluate")
    ap.add_argument("--checkpoint", help="its state_dict")
    ap.add_argument("--baseline", action="store_true", help="evaluate baseline.pt")
    ap.add_argument("--no-int8", action="store_true", help="fp32 only (fast)")
    ap.add_argument("--n-val", type=int, help="evaluate on the first N val images only")
    add_quant_args(ap)
    args = ap.parse_args()

    if args.baseline:
        import os
        model = build_model(BASELINE_CFG)
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baseline.pt"),
                                         map_location="cpu"))
        model.eval()
    else:
        if not (args.config and args.checkpoint):
            ap.error("give --config and --checkpoint, or --baseline")
        model = load_model(args.config, args.checkpoint)

    val = val_batches(args.n_val)
    print(f"fp32 val accuracy: {evaluate(model, val):.4f}")
    if not args.no_int8:
        q = quantize_model(model, args)
        print(f"int8 val accuracy (simulated): {evaluate(q, val, channels_last=True):.4f}")


if __name__ == "__main__":
    main()
