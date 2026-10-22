# What to try

Suggestions, not requirements. Any training procedure that keeps the
architecture rule is fair game.

## Reuse what you built in `pruning_lab/`, `quantization_lab/` and `distillation_lab/`

Each of these earlier projects did one piece of this lab on a small model
(Model B2, Fashion-MNIST). Your copies of them are the best starting point:

**`pruning_lab/`**
- **Channel slimming code.** `python/make_variants.py`, `make_C()`, builds
  variant C by copying the kept filters (rows), the matching input slices of
  the next layer (columns), and the `fc` columns into a smaller model. That
  is exactly `slim()` in `train/compress.py`, for 2 conv layers instead of 6.
  New here: each conv is followed by BatchNorm, whose `weight`, `bias`,
  `running_mean` and `running_var` must be sliced too.
- **Masks don't shrink anything** (Part 1). The 70%-sparse model had the same
  `.pte` size and latency as the dense one. Don't spend time on
  `torch.nn.utils.prune` here.
- **The optional extension** (train the slimmed architecture from scratch)
  is a good comparison point: does pruning beat training
  your final channel list from scratch with the same total budget?

**`quantization_lab/`**
- **The export pipeline.** `python/quantize.py` (PT2E → `CortexMQuantizer` →
  CMSIS-NN → `.pte`) is what `tools/export.py` does, with its
  `--strip-unused` always on.
- **Calibration** (Part B1): tens of class-mixed, normalized images were
  already enough; one class, or unnormalized images, broke the model. The
  same knobs are `export.py`'s `--calib-n`, `--calib-seed` and
  `--calib-file`.
- **Granularity and observers** (Parts B2, B3) are `--weights per-tensor` and
  `--act-observer minmax` here: one experiment each is worth logging.
- **Reading the lowered graph** (Part C2): `export.py` prints the operators.
  Anything `aten::` is a layer running in fp32 on slow portable kernels
  (`--fp32-layers`). Your C2 answer on what an fp32 fallback costs in SRAM
  applies directly to the arena budget.
- **Latency tracks MACs** (Part A): this firmware runs about 2.65 cycles per
  MAC, so `budget.py`'s MAC count predicts latency for your scorecard.

**`distillation_lab/`**
- **The KD loss.** `python/kd.py`'s `kd_loss()` is the loss to write in
  `train/compress.py` (same formula, same T² factor).
- **Choosing T and α** (Part 2): the best values depended on the regime.
  Choose them on your validation split, never the test set.
- **KD after pruning** (Part 6) is this lab's situation, and the result is a
  caution. For the pruned Model B2, KD did *not* beat a plain cross-entropy
  fine-tune, because that model was capacity-bound, while a longer
  fine-tune helped a lot. "Knowledge distillation" below has the check to
  run first.

## Start by measuring

Before pruning anything, work out which budget binds for which layer:

```bash
python3 tools/budget.py                                     # the baseline
python3 tools/budget.py --channels 48 48 96 96 160 160      # a candidate
```

- **Flash** is dominated by the last stage. conv5 alone is 590 KB at int8
  (256 × 256 × 9). Halving the channels of a layer pair cuts the weights
  between them 4×.
- **SRAM (the arena)** is dominated by the first stage: at 32×32, every
  channel of conv0 and conv1 costs 1 KB of activation. Their *sum* must stay
  under ~96 channels. The arena doesn't care how wide stage 3 is.

So the two budgets pull on different ends of the network. Which layers are
cheap to prune for accuracy is a third question, answered by sensitivity
analysis.

Fitting is only the minimum: aim well below the 512 KB Flash budget. Each
halving of the `.pte` earns about 10.5 size points (README.md section 7),
and going below the reference solution's size earns extra credit, so a 256 KB
model that keeps its accuracy beats a 500 KB one.

## Structured pruning is what fits the budgets

`torch.nn.utils.prune` only masks. As in `pruning_lab/`, masked or zeroed
channels don't shrink the `.pte`, the arena or the latency. Only removing
channels does: rebuild the model with a smaller channel list and copy the
kept filters, BatchNorm parameters and running statistics, and the next
layer's input slices into it (`slim()` in `train/compress.py`). Alternatively
use the [Torch-Pruning](https://github.com/VainF/Torch-Pruning) library,
which finds these dependencies automatically. Either way, the result must be
a plain `build_model(cfg)` state_dict.

## Choosing which channels to keep

- **L1/L2 filter norm** (as in `pruning_lab/`): cheap, but blind to BatchNorm, which
  rescales every channel after the conv.
- **BN scale** |γ| (network slimming, Liu et al. 2017): uses the scale BN
  actually applies. It's even better if you first train with an L1 penalty
  on γ, which pushes unimportant channels toward zero.
- **APoZ** (average percentage of zeros after ReLU): needs activations.

Compare at least two, at the same channel counts.

## Per-layer ratios

Run a sensitivity analysis: prune one layer at a time to
several ratios, without fine-tuning, and plot val accuracy. Uniform ratios
are rarely best. Some layers barely notice losing half their channels, while
others fall off a cliff. Remember that the early layers also set the arena.

## Prune gradually

Several prune → fine-tune rounds usually beat one aggressive cut
(`--steps 3` in `train/compress.py`).

## Quantization

`export.py` does post-training quantization (PTQ) with `CortexMQuantizer`:
int8, per-channel symmetric conv weights, per-tensor activations from a
histogram observer. Experiment with the calibration set's size and
representativeness (as in `quantization_lab/` Part B1): `--calib-n`,
`--calib-seed`, `--calib-file`. The other flags (`--act-observer`,
`--weights`) are worth one experiment each.

QAT: the pinned export always does PTQ from your fp32 checkpoint, so QAT can
only help through the weights it leaves behind. Note that ExecuTorch 1.2's
`prepare_qat_pt2e` does **not** work with `CortexMQuantizer` out of the box:
the train-mode graph's BatchNorm counter (`num_batches_tracked += 1`, an
integer add) gets annotated as a quantized add and the first forward pass
fails. If you want QAT-style training, fake-quantize the weights yourself
(e.g. a straight-through `torch.fake_quantize_per_channel_affine` on each
conv weight during fine-tuning). Measure the PTQ gap first (`eval.py`: fp32
vs. int8): it is the most QAT could recover. With per-channel int8 it is
small here (see your own numbers).

## Knowledge distillation

Fine-tune the pruned network with the baseline as teacher:
`loss = α·T²·KL(softmax(t/T) ‖ softmax(s/T)) + (1−α)·CE(s, y)`.

Whether it helps depends on what limits your model, so run the **capacity
check** from `distillation_lab/` first. Compare the pruned model's
training and validation accuracy after a plain fine-tune:

- **Training ≈ validation:** the model is capacity-bound (like the pruned
  Model B2 in `distillation_lab/` Part 6). A better training signal can't add
  capacity, so KD won't help much. Train longer, or keep more channels.
- **Training well above validation:** there is room for a better signal, and
  KD can help. For example, the 50%-pruned model here, after a 15-epoch
  cross-entropy fine-tune, reaches 95.8% on training data but 89.4% on
  validation.

Either way, compare KD with a cross-entropy fine-tune of the same length, and
log both so the comparison is in your experiment log. The teacher's forward
pass makes each epoch about 1.6× slower; a GPU helps.

## Check on "the board" early

Run `tools/run_renode.sh` on your val images once per serious candidate, with
`--config`/`--checkpoint` so it reports how often the board agrees with the
laptop's int8 simulation. A real gap usually means you evaluated something
other than what you exported (another checkpoint, another calibration), a
layer left in fp32, or a preprocessing mismatch (the firmware uses
`data.normalize`; did your training?).

## What won't help here, and why

These are worth understanding (and good material if you write the optional
report):

- **Unstructured sparsity:** the `.pte` is dense, so zeros cost the same
  bytes and cycles as any other weight.
- **k-means weight clustering / codebooks:** the runtime has no codebook
  kernels; weights must be int8 tensors.
- **gzip / Huffman:** the firmware can't decompress, and the model runs in
  place from Flash.
- **Changing the architecture** (depthwise convs, fewer layers): not allowed
  here. It is often the right answer in practice, which is worth a
  sentence in your discussion.

## Time management

Most of the accuracy comes back quickly. For the reference 50%-pruned model
(uniform L1, cross-entropy fine-tune), validation accuracy after
1 / 5 / 10 / 15 epochs was 83.7% / 87.0% / 88.8% / 89.4%. So:

- **Explore cheaply.** Sensitivity scans need no training at all (about 2
  minutes on a CPU). Pareto points and criterion comparisons can use 5–8
  fine-tune epochs, or `--train-subset 15000` (a fixed third of the training
  set) to cut each epoch to a third. Compare candidates under the same budget.
- **Spend on the final candidate.** Gradual pruning (`--steps 3`) plus KD
  and 30 epochs is where the last point or two comes from. Do it once or
  twice, on a GPU.
- **Train on a GPU** (Colab or the course cluster; README.md "Compute"). A
  fine-tune epoch takes roughly 10–20 s there (estimate), against about 75 s
  measured on a 12-core CPU.
  Distillation adds the teacher's forward pass (about 1.6× per epoch).
- **Log every configuration you try** (`tools/log_experiment.py add`): the
  experiment log needs at least five, and it's how you'll pick your final one.
- **Renode** runs at ~2–3 s per image per instance: check 500 images on 4
  instances (~6 min) rather than the whole val split, and only for serious
  candidates. The laptop's int8 simulation (`eval.py`) agrees with the board
  on ~99% of images, so use it for everything else.

## Learning resources (PyTorch)

Tutorials and references for the techniques in this lab. Where a tool does
something different from what this lab needs, the note says so.

**Pruning**
- PyTorch pruning tutorial,
  <https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html>:
  how `torch.nn.utils.prune` works. It applies **masks**, so it explains
  pruning criteria well but shrinks nothing on its own. Useful to understand,
  not to fit the budgets.
- Torch-Pruning, <https://github.com/VainF/Torch-Pruning>: a library that
  physically removes channels, finding the layers that must shrink together
  (its DepGraph). An alternative to writing `slim()` yourself; its tutorials
  and notebooks show structured pruning end to end.
- Li et al., "Pruning Filters for Efficient ConvNets" (ICLR 2017),
  <https://arxiv.org/abs/1608.08710>: L1-norm filter pruning and per-layer
  sensitivity, including a VGG on CIFAR-10. The basis for the `l1`
  criterion.
- Liu et al., "Learning Efficient Convolutional Networks through Network
  Slimming" (ICCV 2017), <https://arxiv.org/abs/1708.06519>: channel
  selection by BatchNorm scale |γ|, the `bn` criterion.

**Quantization**
- torchao, "PyTorch 2 Export Post Training Quantization",
  <https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_ptq.html>:
  the export → `prepare_pt2e` → calibrate → `convert_pt2e` flow that
  `tools/export.py` runs. Its QAT companion,
  <https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_qat.html>,
  explains QAT, but doesn't work with `CortexMQuantizer` in this ExecuTorch
  version (section "Quantization" above).
- ExecuTorch's Cortex-M backend: `backends/cortex_m/README.md` and the
  notebook `examples/arm/cortex_m_mv2_example.ipynb` in your ExecuTorch
  checkout (`~/executorch/executorch`): the lowering `export.py` uses, on
  MobileNetV2.
- Jacob et al., "Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference" (CVPR 2018),
  <https://arxiv.org/abs/1712.05877>: the integer-only scheme (scales, zero
  points, requantization) CMSIS-NN implements.

**Knowledge distillation**
- PyTorch knowledge distillation tutorial,
  <https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html>:
  CIFAR-10, a 1.19 M-parameter teacher and a smaller student. It covers the
  soft-target loss used here and two feature-based variants (hidden
  representations, intermediate feature maps) you could try.
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015),
  <https://arxiv.org/abs/1503.02531>: the original paper (the T² factor is
  explained in section 2).

**All of it, as a course**
- MIT 6.5940, "TinyML and Efficient Deep Learning Computing" (Song Han),
  <https://hanlab.mit.edu/courses/2024-fall-65940>: lecture videos on
  pruning, quantization and distillation, and labs as Colab notebooks
  (Lab 1 pruning, Lab 2 quantization). Its pruning and quantization
  lectures are the most direct companion to this lab.
