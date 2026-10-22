# Training on your laptop's GPU

Optional. Your laptop's CPU trains everything in this lab as is, just slowly
(README.md, "Compute: where to train"). If your laptop has a suitable GPU,
this file shows how to check it and how to set it up.

Only **training** runs on the GPU: `train/compress.py` (and
`train_baseline.py`, which you don't need). Exporting, `eval.py`'s int8
simulation, `check_submission.py` and Renode stay on the CPU, in the course's
ExecuTorch venv.

## 1. Test your laptop

From the lab directory, in the environment you'll train in:

```bash
python3 tools/check_gpu.py
# torch 2.11.0+cpu | CUDA build: None | training device: cpu
# No usable GPU: this PyTorch is a CPU-only build. ...
```

| `training device` | Meaning |
|---|---|
| `cuda` | an NVIDIA GPU, ready to use |
| `mps` | an Apple Silicon GPU (M1 or later), ready to use |
| `cpu` | no usable GPU in this environment |

**In the course's ExecuTorch venv (`~/executorch/et-env`) the answer is
always `cpu`**, whatever your hardware. That venv has a CPU-only PyTorch
(`torch 2.x+cpu`, `CUDA build: None`). Don't reinstall PyTorch there:
ExecuTorch is pinned to its version, and replacing it breaks the export.
Instead, make a second venv just for training (section 2). Training needs
only PyTorch and NumPy, not ExecuTorch.

## 2. Set up a training venv

### NVIDIA GPU (Linux, or Windows with WSL)

First check that the driver works: `nvidia-smi` should list your GPU and,
top right, the highest CUDA version the driver supports. Under WSL, this
needs the NVIDIA Windows driver (recent drivers include WSL support), and
`nvidia-smi` must work *inside* WSL. Then:

```bash
python3 -m venv ~/lab3-train
source ~/lab3-train/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # a CUDA version <= nvidia-smi's
pip install numpy
```

The `cu126` part selects PyTorch built for CUDA 12.6; the selector at
<https://pytorch.org/get-started/locally/> lists the current choices.

### Apple Silicon (M1 or later)

```bash
python3 -m venv ~/lab3-train
source ~/lab3-train/bin/activate
pip install torch numpy
```

The standard macOS PyTorch wheels include the `mps` backend.

Now run the test in section 1 inside `~/lab3-train`. It should print `cuda`
or `mps`.

## 3. Use it

Train in the training venv and do everything else in the ExecuTorch venv,
from the same lab directory. Both use the same `data/` (CIFAR-10), the same
`baseline.pt` and the same `runs/`, so nothing needs copying:

```bash
# Terminal 1: training (GPU)
source ~/lab3-train/bin/activate
python3 train/compress.py prune --channels 32 32 64 64 128 128 --epochs 15 --out runs/first
#   -> "start: ... (training on cuda)"

# Terminal 2: evaluate, export, check, run on the emulated board (CPU)
source ~/executorch/et-env/bin/activate
python3 tools/eval.py --config runs/first/config.json --checkpoint runs/first/checkpoint.pt
python3 tools/export.py --config runs/first/config.json --checkpoint runs/first/checkpoint.pt \
                        --out runs/first/submission.pte
```

The scripts pick the GPU automatically. `--device cpu` (or `cuda`, `mps`)
overrides the choice. Checkpoints are always saved as CPU tensors, so
`export.py` loads them the same way wherever they were trained.
`train/compress.py sensitivity` doesn't train, so it runs on the CPU.

## If it doesn't work

- **`training device: cpu` in `~/lab3-train`, but `nvidia-smi` works:**
  `CUDA build: None` means pip installed the CPU-only PyTorch. Reinstall with
  the `--index-url` line above.
- **`nvidia-smi` not found, or fails, under WSL:** update the NVIDIA driver
  on Windows (not inside WSL), then restart WSL (`wsl --shutdown`).
- **`mps` on a Mac, but training is no faster than the CPU:** small
  models on MPS can be limited by per-batch overhead. Compare one epoch with
  `--device cpu` and keep whichever is faster.
- **Out of GPU memory:** unlikely with this model (well under 1 GB at batch
  128). Close other GPU programs.

If your laptop has no suitable GPU, use the CPU (slow but fine) or Colab
(README.md).
