"""
check_submission.py -- local pre-flight: everything the grader's gate checks
before it runs your model, so you never lose points to a surprise.

    python3 tools/check_submission.py                       # checks <repo>/submission/
    python3 tools/check_submission.py --renode 50           # ...and runs 50 val images under Renode
    python3 tools/check_submission.py path/to/submission

Checks (the gate in README.md; any FAIL scores 0):
  files        submission.pte, config.json, checkpoint.pt, export_flags.json,
               and train/ with your code
  config       config.json follows the architecture rule (model.validate_cfg)
  checkpoint   checkpoint.pt loads into model.build_model(config), strictly
  size         submission.pte <= 524,256 bytes (512 KB region - 32 B header)
  arena        planned memory <= 98,304 bytes (96 KB), one planned buffer
  io           input float [1,3,32,32], output float [1,10]
  ops          every operator is one the firmware has a kernel for
  structure    re-exporting checkpoint.pt with export.py (and your
               export_flags.json) gives the same operators and tensor shapes
  renode       (with --renode N) the firmware loads the model and classifies
               N validation images with no LOAD_ERR / FAULT / timeout

Then, separately (scored, but not part of the gate): the experiment log,
submission/experiments.json, against its 15-point rubric
(tools/log_experiment.py).
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)          # the lab repo root (tools/ is inside it)
sys.path.insert(0, HERE)

PTE_BUDGET_BYTES = 512 * 1024 - 32
ARENA_BUDGET_BYTES = 96 * 1024

# Operators the lab firmware registers (firmware/README: the ExecuTorch
# Cortex-M backend's cortex_m:: ops, plus the portable fp32 kernels an
# unquantized layer of this network can lower to).
FIRMWARE_OPS = {
    "cortex_m::quantize_per_tensor.out", "cortex_m::dequantize_per_tensor.out",
    "cortex_m::quantized_add.out", "cortex_m::quantized_mul.out",
    "cortex_m::minimum.out", "cortex_m::maximum.out",
    "cortex_m::quantized_linear.out", "cortex_m::softmax.out", "cortex_m::transpose.out",
    "cortex_m::pad.out", "cortex_m::quantized_conv2d.out",
    "cortex_m::quantized_depthwise_conv2d.out", "cortex_m::quantized_transpose_conv2d.out",
    "cortex_m::quantized_avg_pool2d.out", "cortex_m::quantized_max_pool2d.out",
    "cortex_m::quantized_batch_matmul.out",
    "aten::convolution.out", "aten::_native_batch_norm_legit_no_training.out",
    "aten::relu.out", "aten::max_pool2d_with_indices.out", "aten::mean.out",
    "aten::addmm.out", "aten::permute_copy.out", "aten::view_copy.out",
    "aten::adaptive_avg_pool2d.out", "aten::batch_norm.out", "aten::mul.out", "aten::add.out",
    "dim_order_ops::_clone_dim_order.out", "dim_order_ops::_to_dim_order_copy.out",
}

REQUIRED_FILES = ["submission.pte", "config.json", "checkpoint.pt", "export_flags.json"]


class Check:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail):
        self.rows.append({"check": name, "ok": bool(ok), "detail": detail})
        return ok

    @property
    def ok(self):
        return all(r["ok"] for r in self.rows)

    def print(self):
        for r in self.rows:
            print(f"  [{'PASS' if r['ok'] else 'FAIL'}] {r['check']:10s} {r['detail']}")


def _io_ok(plan):
    """(ok, detail) for the program's input/output tensors."""
    vals = plan.values
    ins = [vals[i].val for i in plan.inputs]
    outs = [vals[i].val for i in plan.outputs]
    if len(ins) != 1 or len(outs) != 1:
        return False, f"{len(ins)} inputs / {len(outs)} outputs; expected 1 / 1"
    i, o = ins[0], outs[0]
    FLOAT = 6
    if int(i.scalar_type) != FLOAT or list(i.sizes) != [1, 3, 32, 32]:
        return False, f"input is {list(i.sizes)} dtype {int(i.scalar_type)}; expected float [1, 3, 32, 32]"
    if int(o.scalar_type) != FLOAT or list(o.sizes) != [1, 10]:
        return False, f"output is {list(o.sizes)} dtype {int(o.scalar_type)}; expected float [1, 10]"
    return True, "input float [1, 3, 32, 32], output float [1, 10]"


def check_static(sub, structure=True, threads=None, calib=None):
    """All checks except Renode. Returns (Check, info dict). calib: optional
    calibration tensor for the structural re-export (default: 32 training
    images from data.py)."""
    import torch
    from export import (STRUCTURAL_FLAGS, add_quant_args, describe, export_pte,
                        instruction_signature, load_plan)
    from model import build_model, load_cfg

    c = Check()
    info = {}
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(sub, f))]
    train_dir = os.path.join(sub, "train")
    has_train = os.path.isdir(train_dir) and any(
        f.endswith((".py", ".ipynb")) for _, _, fs in os.walk(train_dir) for f in fs)
    detail = []
    if missing:
        detail.append(f"missing {', '.join(missing)}")
    if not has_train:
        detail.append("train/ has no .py or .ipynb")
    c.add("files", not detail, "; ".join(detail) or "all present")
    if missing:
        return c, info

    try:
        cfg = load_cfg(os.path.join(sub, "config.json"))
        c.add("config", True, f"channels {cfg['channels']}")
    except (ValueError, json.JSONDecodeError) as e:
        c.add("config", False, str(e))
        return c, info
    info["channels"] = cfg["channels"]

    model = build_model(cfg)
    try:
        model.load_state_dict(torch.load(os.path.join(sub, "checkpoint.pt"), map_location="cpu"))
        model.eval()
        c.add("checkpoint", True, "loads into build_model(config)")
    except Exception as e:  # noqa: BLE001
        c.add("checkpoint", False, f"does not load into build_model(config): {str(e)[:300]}")
        model = None

    with open(os.path.join(sub, "submission.pte"), "rb") as f:
        pte = f.read()
    try:
        d = describe(pte)
        plan = load_plan(pte)
    except Exception as e:  # noqa: BLE001
        c.add("size", False, f"submission.pte is not a valid ExecuTorch program: {e}")
        return c, info
    info.update(pte_bytes=d["pte_bytes"], planned_bytes=d["planned_bytes"], ops=d["ops"])

    c.add("size", d["pte_bytes"] <= PTE_BUDGET_BYTES,
          f"{d['pte_bytes']:,} bytes (budget {PTE_BUDGET_BYTES:,})")
    n_buf = len([b for b in plan.non_const_buffer_sizes[1:]])
    c.add("arena", d["planned_bytes"] <= ARENA_BUDGET_BYTES and n_buf == 1,
          f"{d['planned_bytes']:,} bytes planned (budget {ARENA_BUDGET_BYTES:,})"
          + ("" if n_buf == 1 else f"; {n_buf} planned buffers, firmware supports 1"))
    c.add("io", *_io_ok(plan))
    unknown = sorted(set(d["ops"]) - FIRMWARE_OPS)
    c.add("ops", not unknown,
          f"no kernel in the firmware for: {', '.join(unknown)}" if unknown
          else f"{len(d['ops'])} operators, all in the firmware"
          + (f" (fp32: {', '.join(d['fp32_ops'])})" if d["fp32_ops"] else ""))

    if structure and model is not None:
        with open(os.path.join(sub, "export_flags.json")) as f:
            flags = json.load(f)
        qargs = add_quant_args(argparse.ArgumentParser()).parse_args([])
        for k in STRUCTURAL_FLAGS:
            if k in flags:
                setattr(qargs, k, flags[k])
        qargs.calib_n = 32          # structure does not depend on calibration
        if threads:
            torch.set_num_threads(threads)
        try:
            _, prog = export_pte(model, qargs, calib=calib)
            ref, got = instruction_signature(prog.buffer), instruction_signature(pte)
            if ref == got:
                c.add("structure", True, f"re-export of checkpoint.pt matches ({len(got)} instructions)")
            else:
                diff = next((k for k in range(min(len(ref), len(got))) if ref[k] != got[k]),
                            min(len(ref), len(got)))
                c.add("structure", False,
                      f"re-export of checkpoint.pt differs from submission.pte at instruction {diff}: "
                      f"expected {ref[diff] if diff < len(ref) else 'end'}, "
                      f"got {got[diff] if diff < len(got) else 'end'} -- export the .pte from this "
                      f"checkpoint/config with export.py")
        except Exception as e:  # noqa: BLE001
            c.add("structure", False, f"re-export failed: {str(e)[:300]}")
    return c, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", nargs="?", default=os.path.join(REPO, "submission"))
    ap.add_argument("--renode", type=int, metavar="N", default=0,
                    help="also run N validation images under Renode")
    ap.add_argument("--jobs", type=int, default=2, help="Renode instances for --renode")
    ap.add_argument("--no-structure", action="store_true", help="skip the (slower) re-export check")
    args = ap.parse_args()

    print(f"Checking {args.submission}")
    c, info = check_static(args.submission, structure=not args.no_structure)
    if args.renode and c.ok:
        import tempfile
        from data import load_split
        from pack_model import pack
        from send_images import DeviceError, run_on_renode
        _, (xva, yva) = load_split()
        with open(os.path.join(args.submission, "submission.pte"), "rb") as f:
            blob = pack(f.read())
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp.write(blob)
        try:
            boot, res = run_on_renode(tmp.name, xva[:args.renode].numpy(), jobs=args.jobs)
            preds = [res[i][0] for i in range(args.renode)]
            bad = [i for i in range(args.renode) if res[i][2] != "OK"]
            acc = sum(int(p == y) for p, y in zip(preds, yva[:args.renode].tolist())) / args.renode
            c.add("renode", not bad, f"{args.renode} images, accuracy {acc:.3f}"
                  + (f"; {len(bad)} not OK" if bad else ""))
        except DeviceError as e:
            c.add("renode", False, str(e))
        finally:
            os.unlink(tmp.name)
    c.print()
    from log_experiment import print_score, score_log
    print("\nExperiment log (15 points, not part of the gate):")
    print_score(*score_log(args.submission))
    if c.ok:
        print("\nAll checks passed." + ("" if args.renode else
                                        " (Run with --renode 50 to also test it on the emulated board.)"))
    else:
        print("\nFix the FAILs above: the grader's gate would score this submission 0.")
        sys.exit(1)


if __name__ == "__main__":
    main()
