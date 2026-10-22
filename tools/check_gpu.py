"""
check_gpu.py -- can this environment train on a GPU?

    python3 tools/check_gpu.py

Prints the PyTorch build and the device train/compress.py and
train_baseline.py would use (the same pick_device() they call). Run it in
the environment you intend to train in. README-local-gpu.md explains the
answers and how to set up a GPU training environment.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from train_baseline import pick_device  # noqa: E402


def main():
    dev = pick_device()
    print(f"torch {torch.__version__} | CUDA build: {torch.version.cuda} | training device: {dev}")
    if dev.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} -- ready to train on it.")
    elif dev.type == "mps":
        print("Apple Silicon GPU (MPS) -- ready to train on it.")
    elif torch.version.cuda is None:
        print("No usable GPU: this PyTorch is a CPU-only build. If you have an NVIDIA GPU, "
              "see README-local-gpu.md section 2.")
    else:
        print("No usable GPU: this PyTorch supports CUDA, but no GPU is visible "
              "(check `nvidia-smi`; README-local-gpu.md).")


if __name__ == "__main__":
    main()
