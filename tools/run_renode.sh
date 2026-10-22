#!/bin/bash
# run_renode.sh -- run a .pte on the emulated board (Renode) with the lab
# firmware, on your validation images: the same harness the grader uses on
# its hidden test set.
#
#   tools/run_renode.sh                                  # submission/submission.pte, 200 val images
#   tools/run_renode.sh runs/x/submission.pte 1000 4     # 1000 images on 4 Renode instances
#   tools/run_renode.sh runs/x/submission.pte 500 4 --config runs/x/config.json --checkpoint runs/x/checkpoint.pt
#
# Arguments: [pte] [n_val_images] [renode_instances] [extra send_images.py flags...]
# Renode runs ~2-3 s per image per instance for a 50%-pruned model; each
# instance uses one CPU core. Needs the ExecuTorch venv active. Paths are
# relative to the lab repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
PTE="${1:-submission/submission.pte}"
N="${2:-200}"
JOBS="${3:-2}"
shift $(( $# < 3 ? $# : 3 ))
exec python3 tools/send_images.py --pte "$PTE" --val-n "$N" --jobs "$JOBS" "$@"
