#!/bin/bash
#SBATCH -J scene_simba
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH -N1 -n1 -c12
#SBATCH --mem=53G
#SBATCH --time=1-00:00:00
#SBATCH -o logs/slurm-simba-%j.out
#
# SIMBA (CPU, PyTorch-BigGraph with 12 workers) + TCR, after ./setup_simba_tcr.sh.
# Two halves in parallel on two workstations (QoS: 2 jobs, 24 CPUs, 110 000 MB):
#   sbatch --export=ALL,HALF=A run_simba.sbatch    # SIMBA on full NeurIPS + PBMC (largest graphs)
#   sbatch --export=ALL,HALF=B run_simba.sbatch    # everything else (cortex, HCA, SIM1/2, GR, stem cells, TCR)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export DEVICE=cuda PBMC_MODE=subset WITH_SIMBA_TCR=1 QUEUE_TAG="simba-$HALF"
unset MINIBATCH_ALL SCENE_CELL_BATCH_SIZE SCENE_RECON_CHUNK

A_RE='neurips_cite_gex\.h5ad|pbmc'
SKIP_RE='SCENE_[0-9]+D_seed[234]|render_notebooks|write_paper_tables'
if [ "$HALF" = A ]; then
  ARGS=(--include "$A_RE" --exclude "$SKIP_RE")
else
  ARGS=(--exclude "$A_RE|$SKIP_RE")
fi

echo "$(date '+%F %T') job $SLURM_JOB_ID on $(hostname), HALF=$HALF, $(nproc) CPUs"
.venv/bin/python run_queue.py "${ARGS[@]}"
