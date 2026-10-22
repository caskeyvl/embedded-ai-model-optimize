"""
log_experiment.py -- PINNED. Keep the experiment log the grader scores
(submission/experiments.json, 15 points; README.md section 7).

    # After training (and exporting) a candidate: measure it and log it
    python3 tools/log_experiment.py add runs/first
    python3 tools/log_experiment.py add runs/first --recipe "uniform 50%, L1, 15 ep CE"

    # Mark the candidate you submit (once it has an on-device result, below)
    python3 tools/log_experiment.py add runs/best --final

    # Log a sensitivity scan (the file train/compress.py sensitivity writes)
    python3 tools/log_experiment.py sensitivity runs/sensitivity_l1.json

    # What would the log score? (check_submission.py prints this too)
    python3 tools/log_experiment.py check

You never edit the JSON by hand: `add` measures everything itself.

`add runs/<name>` needs, in that directory: config.json and checkpoint.pt
(train/compress.py writes both) and submission.pte (tools/export.py
--out runs/<name>/submission.pte). It records the channel list, parameter
and MAC counts, the .pte size and planned arena (read from the .pte), fp32
and simulated-int8 accuracy on the 5,000 validation images (~1 min), and the
recipe. The recipe comes from runs/<name>/log.json (train/compress.py
writes it) or from --recipe. If runs/<name>/renode_val.json exists (from
`tools/run_renode.sh ... --out runs/<name>/renode_val.json`), its on-device
accuracy is recorded too. Logging the same name again replaces its entry. Each entry records when it
was logged (`logged_at`, and `first_logged_at` for its first time).

The rubric (15 points; each item all-or-nothing; full credit is easy if
you log as you go):

  L1  3  the log exists, is valid, and every entry has its measurements
  L2  4  at least 5 candidates with different channel lists or recipes
  L3  3  a sensitivity scan covering conv0..conv5, at least 3 keep ratios each
  L4  3  exactly one candidate is marked final, and it is what you submitted:
         same channels as config.json, same size as submission.pte, and the
         same params, MACs and planned arena recomputed from them
  L5  2  the final candidate has an on-device result on >= 200 val images
"""

import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

LOG_VERSION = 1
DEFAULT_LOG = os.path.join(REPO, "submission", "experiments.json")
MIN_CANDIDATES = 5
MIN_RATIOS = 3
MIN_DEVICE_IMAGES = 200

RUBRIC = [
    ("L1", 3, "the log exists, is valid, and every entry has its measurements"),
    ("L2", 4, f"at least {MIN_CANDIDATES} candidates with different channel lists or recipes"),
    ("L3", 3, f"a sensitivity scan covering conv0..conv5, at least {MIN_RATIOS} keep ratios each"),
    ("L4", 3, "exactly one final candidate, matching config.json and submission.pte"),
    ("L5", 2, f"the final candidate has an on-device result on >= {MIN_DEVICE_IMAGES} val images"),
]
MAX_POINTS = sum(p for _, p, _ in RUBRIC)

CANDIDATE_FIELDS = {"name": str, "channels": list, "recipe": str, "params": int, "macs": int,
                    "pte_bytes": int, "planned_bytes": int, "val_fp32": float, "val_int8": float}


# ---------------------------------------------------------------- read ----

def now_iso():
    """Local time with its UTC offset, e.g. 2026-10-05T14:03:12-06:00."""
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def load_log(path):
    if not os.path.exists(path):
        return {"version": LOG_VERSION, "candidates": [], "sensitivity": []}
    with open(path) as f:
        return json.load(f)


def save_log(log, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=1)


def recipe_from_log_json(run_dir):
    """A one-line recipe from train/compress.py's log.json, if there is one."""
    p = os.path.join(run_dir, "log.json")
    if not os.path.exists(p):
        return None
    a = json.load(open(p)).get("args", {})
    if a.get("cmd") != "prune":
        return None
    parts = [f"{a.get('method', '?')} criterion",
             f"{a.get('steps', 1)} round(s)",
             f"{a.get('epochs', '?')} final epochs at lr {a.get('lr', '?')}"]
    if a.get("steps", 1) > 1:
        parts.append(f"{a.get('epochs_per_step', '?')} epochs per earlier round")
    parts.append(f"KD T={a.get('T')} alpha={a.get('alpha')}" if a.get("kd") else "cross-entropy")
    if a.get("train_subset"):
        parts.append(f"{a['train_subset']} training images")
    if a.get("from_ckpt"):
        parts.append(f"from {a['from_ckpt']}")
    return ", ".join(parts)


# ------------------------------------------------------------- measure ----

def measure(run_dir, threads=None):
    """Everything `add` records about runs/<name>, measured fresh."""
    import argparse as ap_
    import torch
    from eval import evaluate, val_batches
    from export import add_quant_args, describe, quantize_model
    from model import layer_table, load_cfg, load_model

    cfg_path = os.path.join(run_dir, "config.json")
    ck_path = os.path.join(run_dir, "checkpoint.pt")
    pte_path = os.path.join(run_dir, "submission.pte")
    for p, how in ((cfg_path, "train/compress.py writes it"), (ck_path, "train/compress.py writes it"),
                   (pte_path, f"export it first: python3 tools/export.py --config {cfg_path} "
                              f"--checkpoint {ck_path} --out {pte_path}")):
        if not os.path.exists(p):
            sys.exit(f"log_experiment.py: {p} is missing ({how})")
    if threads:
        torch.set_num_threads(threads)

    cfg = load_cfg(cfg_path)
    model = load_model(cfg, ck_path)
    d = describe(open(pte_path, "rb").read())

    # Simulate int8 with the same flags the .pte was exported with.
    qargs = add_quant_args(ap_.ArgumentParser()).parse_args([])
    flags_path = os.path.join(run_dir, "export_flags.json")
    if os.path.exists(flags_path):
        for k, v in json.load(open(flags_path)).items():
            if hasattr(qargs, k):
                setattr(qargs, k, v)
    val = val_batches()
    print(f"measuring {run_dir}: fp32 and int8 accuracy on 5,000 val images ...", flush=True)
    entry = {
        "channels": cfg["channels"],
        "params": sum(p.numel() for p in model.parameters()),
        "macs": sum(r["macs"] for r in layer_table(cfg)),
        "pte_bytes": d["pte_bytes"],
        "planned_bytes": d["planned_bytes"],
        "val_fp32": round(evaluate(model, val), 4),
        "val_int8": round(evaluate(quantize_model(model, qargs), val, channels_last=True), 4),
        "export_flags": {k: getattr(qargs, k) for k in
                         ("calib_n", "calib_seed", "calib_file", "act_observer", "weights", "fp32_layers")},
    }
    dev_path = os.path.join(run_dir, "renode_val.json")
    if os.path.exists(dev_path):
        r = json.load(open(dev_path))
        entry["device"] = {"accuracy": r["accuracy"], "n": r["n"],
                           "agreement": r.get("board_host_agreement"), "source": "renode_val.json"}
    return entry


# --------------------------------------------------------------- score ----

def _valid_candidate(c):
    for k, t in CANDIDATE_FIELDS.items():
        v = c.get(k)
        if t is float:
            if not isinstance(v, (int, float)) or not 0 <= v <= 1:
                return f"{k} must be a number in [0, 1]"
        elif not isinstance(v, t) or (t is int and v <= 0) or (t is str and not v.strip()):
            return f"{k} missing or invalid"
    if len(c["channels"]) != 6 or not all(isinstance(x, int) and x > 0 for x in c["channels"]):
        return "channels must be 6 positive ints"
    return None


def _valid_sensitivity(s):
    layers = s.get("layers")
    if not isinstance(layers, dict):
        return "no layers"
    for i in range(6):
        rows = layers.get(f"conv{i}")
        if not isinstance(rows, list) or not rows:
            return f"conv{i} missing"
        for row in rows:
            if (not isinstance(row, list) or len(row) != 3
                    or not all(isinstance(v, (int, float)) for v in row) or not 0 <= row[2] <= 1):
                return f"conv{i}: rows must be [keep_ratio, channels, accuracy]"
    return None


def _recompute_mismatches(final, cfg, pte_path):
    """The final candidate's params, MACs and planned arena, recomputed from
    config.json and submission.pte, compared with what the log says. These
    are deterministic, so they must match exactly."""
    from export import describe
    from model import build_model, layer_table
    actual = {"params": sum(p.numel() for p in build_model(cfg).parameters()),
              "macs": sum(r["macs"] for r in layer_table(cfg)),
              "planned_bytes": describe(open(pte_path, "rb").read())["planned_bytes"]}
    return [f"{k} {final.get(k)} != {v} recomputed from the submission"
            for k, v in actual.items() if final.get(k) != v]


def score_log(sub_dir, log_path=None):
    """Score submission/experiments.json. Returns (points, rows) with one row
    {"id", "ok", "points", "max_points", "detail"} per rubric item."""
    log_path = log_path or os.path.join(sub_dir, "experiments.json")
    results = {}

    def put(item, ok, detail):
        results[item] = (ok, detail)

    log = None
    if not os.path.exists(log_path):
        put("L1", False, f"{os.path.basename(log_path)} not found (python3 tools/log_experiment.py add ...)")
    else:
        try:
            log = json.load(open(log_path))
            if not isinstance(log, dict) or not isinstance(log.get("candidates"), list) \
                    or not isinstance(log.get("sensitivity"), list):
                raise ValueError("needs a 'candidates' list and a 'sensitivity' list")
        except (ValueError, json.JSONDecodeError) as e:
            put("L1", False, f"not a valid log: {e}")
            log = None
    if log is not None:
        bad = []
        for i, c in enumerate(log["candidates"]):
            problem = _valid_candidate(c) if isinstance(c, dict) else "entry is not an object"
            if problem:
                bad.append((c.get("name", f"#{i}") if isinstance(c, dict) else f"#{i}", problem))
        for i, scan in enumerate(log["sensitivity"]):
            problem = _valid_sensitivity(scan) if isinstance(scan, dict) else "not an object"
            if problem:
                bad.append((f"sensitivity #{i}", problem))
        if not log["candidates"]:
            put("L1", False, "no candidates logged")
        elif bad:
            put("L1", False, f"{bad[0][0]}: {bad[0][1]}")
        else:
            put("L1", True, f"{len(log['candidates'])} candidates, {len(log['sensitivity'])} sensitivity scan(s)")

        cands = [c for c in log["candidates"] if isinstance(c, dict) and not _valid_candidate(c)]
        distinct = {(tuple(c["channels"]), c["recipe"].strip().lower()) for c in cands}
        put("L2", len(distinct) >= MIN_CANDIDATES,
            f"{len(distinct)} distinct candidate(s) (need {MIN_CANDIDATES})")

        full = [s for s in log["sensitivity"] if isinstance(s, dict) and not _valid_sensitivity(s)
                and all(len({r[0] for r in s["layers"][f"conv{i}"]}) >= MIN_RATIOS for i in range(6))]
        put("L3", bool(full), f"{len(full)} complete scan(s)" if full else
            f"need a scan of conv0..conv5 with >= {MIN_RATIOS} keep ratios each "
            "(python3 tools/log_experiment.py sensitivity <file>)")

        finals = [c for c in cands if c.get("final") is True]
        final = finals[0] if len(finals) == 1 else None
        if final is None:
            put("L4", False, f"{len(finals)} candidates marked final (need exactly 1: add ... --final)")
        else:
            problems = []
            try:
                cfg = json.load(open(os.path.join(sub_dir, "config.json")))
                if cfg.get("channels") != final["channels"]:
                    problems.append(f"channels {final['channels']} != config.json {cfg.get('channels')}")
            except (OSError, ValueError):
                problems.append("config.json unreadable")
            pte = os.path.join(sub_dir, "submission.pte")
            if not os.path.exists(pte) or os.path.getsize(pte) != final["pte_bytes"]:
                problems.append(f"pte_bytes {final['pte_bytes']:,} != submission.pte "
                                f"{os.path.getsize(pte) if os.path.exists(pte) else 'missing'}")
            if not problems:
                problems += _recompute_mismatches(final, cfg, pte)
            put("L4", not problems, "; ".join(problems) or f"'{final['name']}' matches the submission")
        dev = (final or {}).get("device") or {}
        ok5 = (final is not None and isinstance(dev.get("n"), int) and dev["n"] >= MIN_DEVICE_IMAGES
               and isinstance(dev.get("accuracy"), (int, float)) and 0 <= dev["accuracy"] <= 1)
        put("L5", ok5, f"on device: {dev['accuracy']:.4f} on {dev['n']} images" if ok5 else
            f"final candidate needs an on-device result on >= {MIN_DEVICE_IMAGES} val images "
            "(tools/run_renode.sh <pte> 200 4 --out runs/<name>/renode_val.json, then add again)")

    rows, total = [], 0
    for item, pts, desc in RUBRIC:
        ok, detail = results.get(item, (False, "not scored: fix L1 first"))
        total += pts if ok else 0
        rows.append({"id": item, "ok": ok, "points": pts if ok else 0, "max_points": pts,
                     "name": desc, "detail": detail})
    return total, rows


def print_score(total, rows):
    for r in rows:
        print(f"  [{'PASS' if r['ok'] else 'MISS'}] {r['id']} {r['points']}/{r['max_points']}  "
              f"{r['name']}: {r['detail']}")
    print(f"  experiment log: {total}/{MAX_POINTS}")


# ----------------------------------------------------------------- CLI ----

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=DEFAULT_LOG, help="the log file (default: submission/experiments.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="measure runs/<name> and log it as a candidate")
    a.add_argument("run_dir")
    a.add_argument("--name", help="default: the run directory's name")
    a.add_argument("--recipe", help="how it was made (default: from its log.json)")
    a.add_argument("--final", action="store_true", help="mark it as the candidate you submit")
    a.add_argument("--threads", type=int, default=os.cpu_count())
    s = sub.add_parser("sensitivity", help="log a sensitivity scan (train/compress.py sensitivity --out)")
    s.add_argument("file")
    c = sub.add_parser("check", help="score the log against the rubric")
    c.add_argument("submission", nargs="?", default=os.path.join(REPO, "submission"))
    args = ap.parse_args()

    if args.cmd == "check":
        log_path = args.log if args.log != DEFAULT_LOG else os.path.join(args.submission, "experiments.json")
        print_score(*score_log(args.submission, log_path))
        return

    log = load_log(args.log)
    if args.cmd == "sensitivity":
        scan = json.load(open(args.file))
        problem = _valid_sensitivity(scan)
        if problem:
            sys.exit(f"log_experiment.py: {args.file}: {problem}")
        scan["source"] = os.path.basename(args.file)
        scan["logged_at"] = now_iso()
        log["sensitivity"] = [x for x in log["sensitivity"] if x.get("source") != scan["source"]] + [scan]
        save_log(log, args.log)
        print(f"logged sensitivity scan {args.file} ({scan.get('method', '?')}) -> {args.log}")
        return

    run_dir = args.run_dir.rstrip("/")
    name = args.name or os.path.basename(run_dir)
    recipe = args.recipe or recipe_from_log_json(run_dir)
    if not recipe:
        sys.exit("log_experiment.py: no log.json in that directory: describe the run with --recipe \"...\"")
    entry = {"name": name, "recipe": recipe, **measure(run_dir, args.threads)}
    old = next((c for c in log["candidates"] if c.get("name") == name), None)
    entry["final"] = bool(args.final or (old or {}).get("final", False))
    entry["logged_at"] = now_iso()
    entry["first_logged_at"] = (old or {}).get("first_logged_at") or entry["logged_at"]
    if args.final:
        for c in log["candidates"]:
            c["final"] = False
    log["candidates"] = [c for c in log["candidates"] if c.get("name") != name] + [entry]
    save_log(log, args.log)
    dev = entry.get("device")
    print(f"logged '{name}': {entry['channels']}, .pte {entry['pte_bytes']:,} B, arena "
          f"{entry['planned_bytes']:,} B, val fp32 {entry['val_fp32']:.4f}, int8 {entry['val_int8']:.4f}"
          + (f", device {dev['accuracy']:.4f} ({dev['n']} images)" if dev else ", no on-device result yet")
          + (" [FINAL]" if entry["final"] else ""))
    print(f"{len(log['candidates'])} candidate(s) in {args.log}")


if __name__ == "__main__":
    main()
