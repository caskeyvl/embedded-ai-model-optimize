"""
budget.py -- which budget binds, for which layer? Estimate a config's int8
Flash and SRAM use from its channel list, before training anything.

    python3 tools/budget.py                                  # the baseline
    python3 tools/budget.py --channels 32 32 64 64 128 128
    python3 tools/budget.py --config my_config.json --exact  # + a real export (~10 s)

Estimates, per layer, at int8 (the memory-budgeting method from the Systems
unit):
  weights   9 * c_in * c_out bytes, + 4 bytes of int32 bias per output
            channel (BatchNorm folds into the conv as that bias)
  live act. bytes of the activations alive while the layer runs: its input
            plus its output (int8), the arena's lower bound at that layer
The first layer also has the fp32 input (12 KB) alive: the model quantizes
its own input as its first op.

--exact exports the config with random weights through export.py and
reports the real .pte size and planned arena, the numbers the gate checks.
"""

import argparse

from model import BASELINE_CHANNELS, NUM_CLASSES, layer_table

PTE_BUDGET = 512 * 1024 - 32
ARENA_BUDGET = 96 * 1024
FP32_INPUT = 3 * 32 * 32 * 4


def estimate(cfg):
    rows = layer_table(cfg)
    total_w = 0
    peak = (0, None)
    for r in rows:
        r["w_bytes"] = r["weights"] + 4 * r["bias"]
        total_w += r["w_bytes"]
        live = r["in_act"] + r["out_act"]
        if r["layer"] == "conv0":
            live = max(live, FP32_INPUT + r["in_act"])   # quantize: fp32 in -> int8 out
        r["live"] = live
        if live > peak[0]:
            peak = (live, r["layer"])
    return rows, total_w, peak


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", type=int, nargs=6, default=BASELINE_CHANNELS)
    ap.add_argument("--config", help="read channels from a config.json instead")
    ap.add_argument("--exact", action="store_true", help="also export it (random weights) and measure")
    args = ap.parse_args()
    if args.config:
        from model import load_cfg
        cfg = load_cfg(args.config)
    else:
        cfg = {"channels": list(args.channels), "num_classes": NUM_CLASSES}

    rows, total_w, (peak, peak_layer) = estimate(cfg)
    print(f"channels {cfg['channels']}")
    print(f"{'layer':6s} {'shape':22s} {'MACs':>12s} {'weights (B)':>12s} {'live act. (B)':>14s}")
    for r in rows:
        print(f"{r['layer']:6s} {r['shape']:22s} {r['macs']:>12,} {r['w_bytes']:>12,} {r['live']:>14,}")
    macs = sum(r["macs"] for r in rows)
    print(f"{'total':6s} {'':22s} {macs:>12,} {total_w:>12,}")
    print(f"\nFlash: ~{total_w:,} B of weights + a few KB of program  vs budget {PTE_BUDGET:,}"
          f"  {'OK' if total_w < PTE_BUDGET - 8192 else 'OVER / TIGHT'}")
    print(f"SRAM:  peak live activations ~{peak:,} B at {peak_layer}  vs arena {ARENA_BUDGET:,}"
          f"  {'OK' if peak <= ARENA_BUDGET else 'OVER'}")

    if args.exact:
        import argparse as ap_
        import torch
        from export import add_quant_args, describe, export_pte
        from model import build_model
        torch.manual_seed(0)
        model = build_model(cfg).eval()
        qargs = add_quant_args(ap_.ArgumentParser()).parse_args([])
        _, prog = export_pte(model, qargs, calib=torch.randn(16, 3, 32, 32))
        d = describe(prog.buffer)
        print(f"\nexact (export.py, random weights): .pte {d['pte_bytes']:,} B "
              f"({'OK' if d['pte_bytes'] <= PTE_BUDGET else 'OVER'}), planned arena "
              f"{d['planned_bytes']:,} B ({'OK' if d['planned_bytes'] <= ARENA_BUDGET else 'OVER'})")


if __name__ == "__main__":
    main()
