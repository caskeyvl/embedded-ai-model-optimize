# tools/ — the provided Python tools and scripts

Everything here is provided. Your own code goes in `train/` (start from
`train/compress.py`). Run every tool from the lab's root directory, e.g.
`python3 tools/budget.py`. Paths you pass (`runs/...`, `submission/...`) are
relative to the root, and defaults (`data/`, `baseline.pt`, `submission/`,
`firmware/`) are found from the root wherever you run from.

**Environment.** Everything runs in the course's ExecuTorch venv
(`source ~/executorch/et-env/bin/activate`). The training tools
(`model.py`, `data.py`, `train_baseline.py`, `eval.py`'s fp32 part,
`check_gpu.py`) need only PyTorch and NumPy, so they also run in a separate
GPU training venv or on Colab (README-local-gpu.md).

**May I modify it?** The ones marked **pinned** define how your model is
built, preprocessed, exported, packed and checked. The grader runs its own
unmodified copies, so changing yours only makes your local results differ
from your grade. Use their command-line flags instead. The others are
helpers: change, copy or ignore them freely.

## At a glance

| Tool | What it's for | When you use it | Pinned? |
|---|---|---|---|
| `budget.py` | Per-layer weight bytes, MACs and live activations for a channel list, against both budgets | Before training any candidate (instant) | no |
| `eval.py` | Validation accuracy, fp32 and int8 as the board computes it | After training each candidate | no |
| `export.py` | Checkpoint + config → int8 `.pte` for the board | Each serious candidate, and your final one | **yes** |
| `run_renode.sh` | Run a `.pte` on the emulated board on validation images | Each serious candidate | no |
| `log_experiment.py` | Measures a candidate and adds it to your experiment log (graded, 15 points) | After each candidate; once with `--final` | **yes** |
| `check_submission.py` | Every check the grader's gate runs, a short Renode run, and your log's score | Before you submit | **yes** |
| `model.py` | The network, built from a channel list | Imported by your code | **yes** |
| `data.py` | CIFAR-10 download, train/val split, normalization, augmentation | Imported by your code; run once to download | **yes** (normalization) |
| `train_baseline.py` | How `baseline.pt` was trained; reusable training loop | Imported by your code (you needn't retrain the baseline) | no |
| `check_gpu.py` | Whether this environment can train on a GPU | Once, when setting up | no |
| `send_images.py` | Streams images to Renode or a real board, collects predictions | Through `run_renode.sh`, or directly for a real board | **yes** |
| `flash_board.sh` | Write the firmware and a model to a real board | Only if you have a board (optional) | no |
| `pack_model.py` | Wraps a `.pte` in the 32-byte header the firmware checks | Called for you by the scripts above | **yes** |

## The tools

### `budget.py`: check a channel list against the budgets

```bash
python3 tools/budget.py --channels 32 32 64 64 128 128
python3 tools/budget.py --config runs/x/config.json --exact
```

Prints, per layer, the int8 weight bytes, MACs, and the activation bytes
alive while the layer runs, then the totals against the 512 KB Flash and
96 KB arena budgets. Pure arithmetic, so it's instant: use it to choose a
channel list before spending time on training. `--exact` also exports the
config with random weights and reports the true `.pte` size and planned
arena (~10 s).

### `eval.py`: validation accuracy, fp32 and int8

```bash
python3 tools/eval.py --config runs/x/config.json --checkpoint runs/x/checkpoint.pt
python3 tools/eval.py --baseline
```

Reports accuracy on your 5,000-image validation split for the fp32 PyTorch
model, and for the int8 model `export.py` would produce, simulated in
PyTorch. The int8 number agrees with the board on ~99% of images, so it's
the number to compare candidates by. Takes the same quantization flags as
`export.py`. `--no-int8` skips the int8 part (fp32 only, any environment);
`--n-val N` uses the first N images. Your code can import `evaluate()` and
`val_batches()`.

### `export.py` (pinned): make the `.pte`

```bash
python3 tools/export.py --config runs/x/config.json --checkpoint runs/x/checkpoint.pt \
                        --out runs/x/submission.pte
```

The only way to make a `.pte` you can submit: post-training int8
quantization with the ExecuTorch Cortex-M quantizer, lowered to CMSIS-NN
kernels. Prints the `.pte` size and planned arena against the budgets. Also
writes `export_flags.json` next to the `.pte` (submit it). Flags you may
change (calibration data, observer, weight granularity, fp32 layers) are in
README.md section 4, "What `export.py` lets you change".

### `run_renode.sh`: run on the emulated board

```bash
tools/run_renode.sh runs/x/submission.pte 500 4 \
    --config runs/x/config.json --checkpoint runs/x/checkpoint.pt
```

Arguments: `.pte`, number of validation images, number of parallel Renode
instances (one CPU core each), then any `send_images.py` flags. Starts the
emulated STM32F407 with the lab firmware, streams the images over its UART
and prints on-device accuracy, per-class accuracy and, with
`--config`/`--checkpoint`, how often the board agrees with `eval.py`'s int8
simulation. About 2–3 s per image per instance. This is the same harness the
grader uses on its hidden test set.

### `log_experiment.py` (pinned): the experiment log

```bash
python3 tools/log_experiment.py add runs/x                 # after exporting runs/x/submission.pte
python3 tools/log_experiment.py add runs/x --final         # the candidate you submit
python3 tools/log_experiment.py sensitivity runs/sensitivity_l1.json
python3 tools/log_experiment.py check                      # your score against the rubric
```

Keeps `submission/experiments.json`, which is worth 15 points (rubric in
README.md section 7). `add` measures the run itself: channels, parameters,
MACs, `.pte` size and arena (from `runs/x/submission.pte`), and fp32 and
int8 accuracy on all 5,000 validation images (about a minute). It takes the
recipe from `runs/x/log.json` (`train/compress.py` writes it) or from
`--recipe "..."`, and the on-device result from `runs/x/renode_val.json` if
you ran `run_renode.sh ... --out runs/x/renode_val.json`. Adding a name again
replaces its entry. `sensitivity` adds a scan written by
`train/compress.py sensitivity --out ...` (same format if you write your own:
`{"layers": {"conv0": [[keep_ratio, channels, accuracy], ...], ...}}`).

### `check_submission.py` (pinned): before you submit

```bash
python3 tools/check_submission.py --renode 50
python3 tools/check_submission.py path/to/submission
```

Runs every check of the grader's gate on `submission/`: required files,
architecture rule, checkpoint loads, `.pte` size, planned arena, operators,
and that your `.pte` really was exported from your checkpoint. `--renode N`
also classifies N validation images on the emulated board. Anything marked
FAIL would score 0, so fix it before submitting. It then prints your
experiment log's score (not part of the gate).

### `model.py` (pinned): the network

`build_model(cfg)` builds the network from
`{"channels": [6 ints], "num_classes": 10}`; `load_model(config, checkpoint)`
rebuilds a trained one. `layer_table(cfg)` gives per-layer sizes (what
`budget.py` prints). Layers are `convs[i]` / `bns[i]` (i = 0..5) and `fc`,
which is what channel surgery indexes. `python3 tools/model.py` prints the
baseline's layers.

### `data.py` (pinned): CIFAR-10

`python3 tools/data.py` downloads CIFAR-10 (~170 MB) into `data/` and prints
the split. Your code uses `loaders()` (augmented training batches,
validation batches; `train_subset=N` for quick runs), `load_split()`,
`normalize()` and `calibration_set()`. **The normalization is pinned**: the
firmware normalizes every frame with the same constants, so train with
them.

### `train_baseline.py`: the baseline's recipe and a reusable training loop

You don't need to run it: `baseline.pt` is its output (30 epochs, 92.1%
val). Your code reuses `train_one_epoch()` (with an optional `loss_fn`
for distillation), `make_optimizer()`, `pick_device()` (GPU if available)
and `cpu_state_dict()` (save checkpoints that `export.py` can load).

### `check_gpu.py`: can this environment use a GPU?

```bash
python3 tools/check_gpu.py
```

Prints the PyTorch build and the device training would use (`cuda`, `mps`
or `cpu`). In the ExecuTorch venv it always says `cpu`; see
README-local-gpu.md.

### `send_images.py` (pinned): talk to the board

`run_renode.sh` and `check_submission.py` use it for you. Use it directly
for a real board:

```bash
python3 tools/send_images.py --port /dev/ttyUSB0 --val-n 500
```

It frames each image (sync bytes, index, length, RGB pixels, CRC), sends it,
and reads back one prediction line. With `--pte` (and no `--port`) it starts
Renode itself; `--jobs N` runs N instances in parallel; `--out file.json`
saves per-image results. On a real board it resets the board through its
ST-LINK and also reports latency. The protocol is described in
`firmware/README.md` (optional reading).

### `flash_board.sh`: real board only (optional)

```bash
tools/flash_board.sh runs/x/submission.pte              # firmware + model
tools/flash_board.sh runs/x/submission.pte --model-only # just the model
```

Writes the firmware (once) and your model to a STM32F4DISCOVERY through its
ST-LINK. Needs OpenOCD and a 3.3 V USB-serial adapter on PA2/PA3. You don't
need a board for this lab: Renode gives the same predictions.

### `pack_model.py` (pinned): the model header

Wraps a `.pte` in the 32-byte header (magic, length, CRC-32) the firmware
checks before running it. `run_renode.sh`, `check_submission.py`,
`flash_board.sh` and the grader call it; you never need to.
