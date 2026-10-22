"""
data.py -- CIFAR-10 for this lab: download, split, preprocess, augment.

    python3 tools/data.py          # download ahead of time (~170 MB) and print the split

Downloads the official "python version" archive into <repo>/data/ on first use,
or uses $CIFAR10_DIR if that is set (point it at an existing
cifar-10-batches-py/ to avoid a second download). No torchvision needed.

Split: the 50,000 training images are split 45,000 train / 5,000 validation
with a fixed seed (SPLIT_SEED). Tune everything on the validation split.
The CIFAR-10 *test* set is what the grader's hidden images come from: do not
train or tune on it (honor code). This file deliberately has no loader for it.

Preprocessing is fixed, because the firmware does it, not your model:
images arrive on the board as uint8 RGB, HWC, 3072 bytes, and the firmware
computes (pixel / 255 - MEAN[c]) / STD[c] in float32 and feeds that to the
model. normalize() below is the exact same computation, so the board's input
tensor is bit-for-bit the one PyTorch sees. Train with these constants.
"""

import os
import pickle
import tarfile
import urllib.request
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)          # the lab repo root (tools/ is inside it)
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# CIFAR-10 training-set channel statistics (RGB, on the 0..1 scale). The
# firmware has the same constants compiled in; do not change them.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

SPLIT_SEED = 0
N_VAL = 5000


def data_dir() -> str:
    """The cifar-10-batches-py directory, downloading it if needed."""
    d = os.environ.get("CIFAR10_DIR")
    if d:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"$CIFAR10_DIR={d} does not exist")
        return d
    root = os.path.join(REPO, "data")
    d = os.path.join(root, "cifar-10-batches-py")
    if not os.path.isdir(d):
        os.makedirs(root, exist_ok=True)
        tgz = os.path.join(root, "cifar-10-python.tar.gz")
        if not os.path.exists(tgz):
            print(f"Downloading {URL} (~170 MB) ...")
            urllib.request.urlretrieve(URL, tgz + ".part")
            os.rename(tgz + ".part", tgz)
        with tarfile.open(tgz) as t:
            t.extractall(root)
    return d


def _read_batch(path: str):
    with open(path, "rb") as f, warnings.catch_warnings():
        # NumPy >= 2.4 warns about the archive's old pickle format; harmless.
        warnings.simplefilter("ignore")
        d = pickle.load(f, encoding="bytes")
    # Stored as N x 3072 = N x (1024 R, 1024 G, 1024 B); convert to HWC,
    # the byte order the board receives.
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(x), np.array(d[b"labels"], dtype=np.int64)


def load_train_u8():
    """All 50k training images: (uint8 [N,32,32,3] HWC, int64 [N])."""
    d = data_dir()
    parts = [_read_batch(os.path.join(d, f"data_batch_{i}")) for i in range(1, 6)]
    return (torch.from_numpy(np.concatenate([p[0] for p in parts])),
            torch.from_numpy(np.concatenate([p[1] for p in parts])))


def split_indices():
    """(train_idx, val_idx) into load_train_u8()'s arrays: 45k / 5k."""
    perm = torch.randperm(50000, generator=torch.Generator().manual_seed(SPLIT_SEED))
    return perm[N_VAL:], perm[:N_VAL]


def load_split():
    """((x_train_u8, y_train), (x_val_u8, y_val)), images uint8 HWC."""
    x, y = load_train_u8()
    tr, va = split_indices()
    return (x[tr], y[tr]), (x[va], y[va])


def normalize(x_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [N,32,32,3] HWC -> float32 [N,3,32,32] NCHW, normalized.
    The same float32 operations the firmware performs on each pixel."""
    x = x_u8.to(torch.float32) / 255.0
    x = (x - torch.tensor(MEAN, dtype=torch.float32)) / torch.tensor(STD, dtype=torch.float32)
    return x.permute(0, 3, 1, 2).contiguous()


def augment_u8(x_u8: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Standard CIFAR augmentation on a uint8 HWC batch: random 32x32 crop
    from the image zero-padded by 4 on each side, then a random horizontal
    flip."""
    n = x_u8.shape[0]
    xp = torch.nn.functional.pad(x_u8.permute(0, 3, 1, 2), (4, 4, 4, 4)).permute(0, 2, 3, 1)
    oy = torch.randint(0, 9, (n,), generator=gen)
    ox = torch.randint(0, 9, (n,), generator=gen)
    rows = (oy[:, None] + torch.arange(32))[:, :, None]          # [n,32,1]
    cols = (ox[:, None] + torch.arange(32))[:, None, :]          # [n,1,32]
    out = xp[torch.arange(n)[:, None, None], rows, cols]         # [n,32,32,3]
    flip = torch.rand(n, generator=gen) < 0.5
    out[flip] = out[flip].flip(2)
    return out


class Batches:
    """Iterable of (x float NCHW normalized, y) batches over uint8 images.
    Holds the whole split in memory (150 MB as uint8); no DataLoader workers
    needed. augment=True shuffles and augments every epoch."""

    def __init__(self, x_u8, y, batch_size=128, augment=False, seed=1):
        self.x, self.y = x_u8, y
        self.batch_size = batch_size
        self.augment = augment
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self):
        return (len(self.y) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.y)
        order = torch.randperm(n, generator=self.gen) if self.augment else torch.arange(n)
        for i in range(0, n, self.batch_size):
            idx = order[i:i + self.batch_size]
            xb = self.x[idx]
            if self.augment:
                xb = augment_u8(xb, self.gen)
            yield normalize(xb), self.y[idx]


def loaders(batch_size=128, seed=1, train_subset=None):
    """(train Batches with augmentation, val Batches without).
    train_subset=N trains on a fixed random N of the 45k training images
    (class-mixed, same N images every epoch): for quick exploratory runs.
    Validation always uses all 5k."""
    (xtr, ytr), (xva, yva) = load_split()
    if train_subset is not None and train_subset < len(ytr):
        idx = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(123))[:train_subset]
        xtr, ytr = xtr[idx], ytr[idx]
    return (Batches(xtr, ytr, batch_size, augment=True, seed=seed),
            Batches(xva, yva, 500, augment=False))


def calibration_set(n: int = 256, seed: int = 0) -> torch.Tensor:
    """n training-split images (no augmentation), normalized, for PTQ
    calibration. Random with a fixed seed so that the classes are mixed."""
    (xtr, _), _ = load_split()
    idx = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(seed))[:n]
    return normalize(xtr[idx])


def pixel_lut() -> torch.Tensor:
    """float32 [3,256]: normalize() applied to every uint8 value, per
    channel. The firmware computes the same table at boot."""
    v = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 256, 1).expand(1, 1, 256, 3)
    return normalize(v.contiguous())[0, :, 0, :]


if __name__ == "__main__":
    (xtr, ytr), (xva, yva) = load_split()
    print(f"CIFAR-10 in {data_dir()}: {len(ytr)} train, {len(yva)} val, images {tuple(xtr.shape[1:])} uint8 HWC")
    print("val class counts:", torch.bincount(yva, minlength=10).tolist())
