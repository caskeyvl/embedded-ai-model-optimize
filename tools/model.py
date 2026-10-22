"""
model.py -- the lab's CIFAR-10 classifier, parameterized by its channel list.

    cfg = {"channels": [64, 64, 128, 128, 256, 256], "num_classes": 10}
    model = build_model(cfg)

Every conv is 3x3, padding 1, no bias, followed by BatchNorm and ReLU:

    stage 1   conv(3   -> c0)  conv(c0 -> c1)  maxpool2     32x32 -> 16x16
    stage 2   conv(c1  -> c2)  conv(c2 -> c3)  maxpool2     16x16 ->  8x8
    stage 3   conv(c3  -> c4)  conv(c4 -> c5)  maxpool2      8x8  ->  4x4
    global-avg-pool -> linear(c5 -> 10)

The baseline is channels = [64, 64, 128, 128, 256, 256]: about 1.15 M
parameters, 4.6 MB in fp32 and 1.15 MB in int8. Neither fits the 512 KB
model budget.

A pruned model is the same network with smaller channel counts, so its
architecture is fully described by its channel list (the architecture rule
in README.md). Do not change this file: the grader rebuilds your model from
config.json with an unmodified copy of it.

Layers are kept in two ModuleLists, `convs[i]` and `bns[i]` (i = 0..5), so
that channel surgery can index them directly: removing output channel k of
convs[i] means removing entry k of bns[i] and input channel k of convs[i+1]
(or of `fc` for i = 5).
"""

import json

import torch
import torch.nn as nn

BASELINE_CHANNELS = [64, 64, 128, 128, 256, 256]
NUM_CLASSES = 10
BASELINE_CFG = {"channels": BASELINE_CHANNELS, "num_classes": NUM_CLASSES}

# Spatial size each conv runs at (all convs keep size; pools halve it).
CONV_HW = [32, 32, 16, 16, 8, 8]
# A 2x2 max-pool follows these convs.
POOL_AFTER = {1, 3, 5}


class CifarVGG(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        validate_cfg(cfg)
        ch = list(cfg["channels"])
        self.channels = ch
        c_in = [3] + ch[:-1]
        self.convs = nn.ModuleList(
            nn.Conv2d(c_in[i], ch[i], kernel_size=3, padding=1, bias=False) for i in range(6))
        self.bns = nn.ModuleList(nn.BatchNorm2d(ch[i]) for i in range(6))
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch[5], cfg["num_classes"])

    def forward(self, x):
        for i in range(6):
            x = torch.relu(self.bns[i](self.convs[i](x)))
            if i in POOL_AFTER:
                x = self.pool(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def cfg(self) -> dict:
        return {"channels": list(self.channels), "num_classes": self.fc.out_features}


def validate_cfg(cfg: dict):
    """Raises ValueError unless cfg follows the architecture rule: the
    baseline's six convs, each with between 1 and the baseline's channel
    count, and 10 classes."""
    if not isinstance(cfg, dict) or set(cfg) != {"channels", "num_classes"}:
        raise ValueError('cfg must be exactly {"channels": [...6 ints...], "num_classes": 10}')
    ch = cfg["channels"]
    if not isinstance(ch, list) or len(ch) != 6:
        raise ValueError(f"channels must be a list of 6 ints, got {ch!r}")
    for i, (c, cmax) in enumerate(zip(ch, BASELINE_CHANNELS)):
        if not isinstance(c, int) or isinstance(c, bool) or not 1 <= c <= cmax:
            raise ValueError(f"channels[{i}] = {c!r}: must be an int in [1, {cmax}]")
    if cfg["num_classes"] != NUM_CLASSES:
        raise ValueError(f"num_classes must be {NUM_CLASSES}")


def build_model(cfg: dict = None) -> CifarVGG:
    return CifarVGG(cfg or BASELINE_CFG)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    validate_cfg(cfg)
    return cfg


def load_model(cfg_path_or_dict, checkpoint: str) -> CifarVGG:
    """Rebuild a model from a config (path or dict) and a state_dict file,
    in eval mode."""
    cfg = load_cfg(cfg_path_or_dict) if isinstance(cfg_path_or_dict, str) else cfg_path_or_dict
    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def layer_table(cfg: dict) -> list:
    """Per-layer sizes, computed from the channel list alone.

    Returns one dict per layer (conv0..conv5, fc) with:
      weights   weight count after BatchNorm is folded into the conv
                (the BN becomes a per-channel bias: + c_out)
      macs      multiply-accumulates per inference
      in_act    input activation elements
      out_act   output activation elements (before any pool)
    At int8, bytes = elements (+ 4 bytes per int32 bias)."""
    ch = cfg["channels"]
    c_in = [3] + ch[:-1]
    rows = []
    for i in range(6):
        hw = CONV_HW[i]
        rows.append({
            "layer": f"conv{i}",
            "shape": f"{c_in[i]}->{ch[i]} @ {hw}x{hw}",
            "weights": 9 * c_in[i] * ch[i],
            "bias": ch[i],
            "macs": hw * hw * 9 * c_in[i] * ch[i],
            "in_act": hw * hw * c_in[i],
            "out_act": hw * hw * ch[i],
        })
    rows.append({
        "layer": "fc",
        "shape": f"{ch[5]}->{cfg['num_classes']}",
        "weights": ch[5] * cfg["num_classes"],
        "bias": cfg["num_classes"],
        "macs": ch[5] * cfg["num_classes"],
        "in_act": ch[5],
        "out_act": cfg["num_classes"],
    })
    return rows


if __name__ == "__main__":
    m = build_model()
    print(f"baseline: {count_params(m):,} parameters")
    for r in layer_table(BASELINE_CFG):
        print(f"  {r['layer']:6s} {r['shape']:22s} weights {r['weights']:>8,}  MACs {r['macs']:>11,}")
