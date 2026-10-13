#!/bin/bash
#SBATCH -J scene_ws
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH -N1 -n1 -c12
#SBATCH --mem=53G
#SBATCH --time=1-00:00:00
#SBATCH -o logs/slurm-ws-%j.out
#
# Workstation partition: one dedicated RTX 5000 Ada (32 GB) per node, so SCENE trains on the
# full matrix as in the paper (no mini-batches). QoS allows 2 jobs at once (24 CPUs, 110 000 MB total, so 2 × 53G),
# so the remaining work is split into two disjoint halves that run in parallel:
#   sbatch --export=ALL,HALF=A run_ws.sbatch    # PBMC + NeurIPS stem cells
#   sbatch --export=ALL,HALF=B run_ws.sbatch    # everything else
# Final figures/tables are excluded; run them once both halves are done.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export DEVICE=cuda PBMC_MODE=subset QUEUE_TAG="ws-$HALF"
unset MINIBATCH_ALL SCENE_CELL_BATCH_SIZE SCENE_RECON_CHUNK

A_RE='pbmc|neurips_cite_gex_stem|neurips_cite_stem'
SKIP_RE='SCENE_[0-9]+D_seed[234]|render_notebooks|write_paper_tables'
if [ "$HALF" = A ]; then
  ARGS=(--include "$A_RE" --exclude "$SKIP_RE")
else
  ARGS=(--exclude "$A_RE|$SKIP_RE")
fi

echo "$(date '+%F %T') job $SLURM_JOB_ID on $(hostname), HALF=$HALF"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
.venv/bin/python run_queue.py "${ARGS[@]}"
