#!/bin/bash
#SBATCH -J scene_repro
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N1 -n1 -c8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH -o logs/slurm-%j.out
#
# Runs the resumable queue on one GPU. If the work takes longer than the 8 h limit, each job
# first submits its own continuation (starts after this job ends, for any reason). A job that
# finds nothing left to run does not resubmit, and the chain stops.
#
# Submit from the reproduction/ folder:
#   sbatch run_cluster.sbatch                                    # everything, full PBMC
#   sbatch --export=ALL,PART=scvi run_cluster.sbatch             # only scVI (Mac does the rest)
#   sbatch --export=ALL,PART=cluster run_cluster.sbatch          # scVI + everything PBMC (Mac does the rest)
#   sbatch --export=ALL,PART=rest run_cluster.sbatch             # everything left (dimension sweep: seeds 0-1),
#                                                                  all SCENE fits with mini-batches (shared GPUs)
# Stop the chain:  scancel -n scene_repro
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export DEVICE=cuda
export PART="${PART:-all}"

SCVI_RE='train_scvi\.py|benchmark_embedding\.py .*--embedding_key scVI'
if [ "$PART" = scvi ]; then
  export PBMC_MODE="${PBMC_MODE:-subset}"   # must match the Mac run (same 30k-cell subset)
  ARGS=(--include "$SCVI_RE")
elif [ "$PART" = rest ]; then
  # Shared node: other users' processes hold most GPU memory, so full-matrix SCENE does not fit.
  export PBMC_MODE="${PBMC_MODE:-subset}" MINIBATCH_ALL=1 CELL_BATCH="${CELL_BATCH:-2048}"
  export SCENE_CELL_BATCH_SIZE="$CELL_BATCH" SCENE_RECON_CHUNK="$CELL_BATCH"   # imputation.py patch
  ARGS=(--exclude 'SCENE_[0-9]+D_seed[234]')
elif [ "$PART" = cluster ]; then
  export PBMC_MODE="${PBMC_MODE:-subset}"
  ARGS=(--include "$SCVI_RE|pbmc")
else
  export PBMC_MODE="${PBMC_MODE:-full}"
  ARGS=()
fi

left=$(.venv/bin/python run_queue.py --dry-run "${ARGS[@]}" 2>/dev/null | grep -c '\.sh: ')
echo "$(date '+%F %T') job $SLURM_JOB_ID on $(hostname): $left commands left (PART=$PART, PBMC_MODE=$PBMC_MODE)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "$left" -eq 0 ]; then
  echo "nothing left, chain finished"; exit 0
fi

sbatch --dependency=afterany:"$SLURM_JOB_ID" --export=ALL,PART="$PART",PBMC_MODE="$PBMC_MODE" run_cluster.sbatch
.venv/bin/python run_queue.py "${ARGS[@]}"
