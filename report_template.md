# Lab 3 report: <your name>

> **Optional and not graded.** Your grade comes from the board and from
> `submission/experiments.json` (README.md section 7). A short report like this
> is good practice: it's what you'd hand a colleague. The numbers it needs are
> all in your experiment log.

At most 4 pages including figures. Replace every _italic_ prompt.

## 1. Final model

| | Baseline | Final |
|---|---:|---:|
| Channels | 64 64 128 128 256 256 | _..._ |
| Parameters | 1,148,874 | _..._ |
| `.pte` bytes | 1,167,848 (int8) | _..._ |
| Planned arena bytes | 131,072 (int8) | _..._ |
| Val accuracy, fp32 (PyTorch) | _..._ | _..._ |
| Val accuracy, int8 simulated (`eval.py`) | _..._ | _..._ |
| Val accuracy, on device (`run_renode.sh`, N = _..._) | — | _..._ |
| Board/laptop agreement | — | _..._ |
| Latency, real board (optional) | — | _... ms_ |

_Export flags used (from export_flags.json), and the full recipe in a few
lines: criterion, per-layer ratios, rounds, epochs, learning rates, KD
settings, calibration._

## 2. Accuracy vs. size (Pareto plot)

_A plot of every configuration you tried, at least 5: x = `.pte` bytes (or
int8 parameter count), y = val accuracy (laptop-simulated int8). Mark the 512
KB budget, your final model, and its on-device accuracy. Label points by what
they were (e.g. "uniform 50% L1, CE", "sensitivity, BN, KD")._

## 3. Per-layer sensitivity

_Accuracy vs. fraction of channels kept, one curve per conv layer, pruning
one layer at a time without fine-tuning. Which layers are sensitive? How did
this, together with the Flash and arena budgets, set your per-layer
ratios?_

## 4. Compression scorecard

_What each step of your pipeline bought you. One row per technique, in the
order you applied them, with the numbers measured after that step._

| Step | Flash (`.pte` B) | SRAM (arena B) | Latency / MACs | Val acc. | Notes |
|---|---:|---:|---:|---:|---|
| Baseline, fp32 | (fp32 export) | | 152.8 M MACs | | |
| + int8 PTQ | 1,167,848 | 131,072 | | | |
| + structured pruning (_..._) | | | | | |
| + fine-tuning / KD | (same) | (same) | (same) | | |
| + _..._ | | | | | |

## 5. Discussion

- _What didn't work, and why (with numbers)._
- _The gap between PyTorch and on-device accuracy: how big, and where it
  comes from._
- _Something from README-suggestions.md's "what won't help" list, argued
  with this lab's numbers._
- _What you would do next with another week, or without the architecture
  rule._
