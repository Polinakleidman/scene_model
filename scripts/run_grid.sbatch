#!/bin/bash
#SBATCH -J mm_grid
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH -N1 -n1 -c12
#SBATCH --mem=53G
#SBATCH --time=1-00:00:00
#SBATCH -o logs/grid-%j.out
#
# Runs one worker's command list on a dedicated workstation GPU.
#   sbatch --export=ALL,WORKER=0 scripts/run_grid.sbatch
#   sbatch --export=ALL,WORKER=1 scripts/run_grid.sbatch
#   sbatch --export=ALL,GRID=weights,WORKER=0 scripts/run_grid.sbatch   # another command list
# Each run is skipped if its metrics.json exists, so resubmitting continues where it stopped.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export PATH="$HOME/reproduction/.venv/bin:$PATH"   # same environment as the reproduction
LIST="experiments/${GRID:-grid}_worker${WORKER}.txt"   # GRID=weights -> experiments/weights_worker*.txt
echo "$(date '+%F %T') worker $WORKER on $(hostname): $(wc -l < "$LIST") commands"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
i=0
while read -r cmd; do
  i=$((i + 1))
  echo "$(date '+%F %T') [$i] START $cmd"
  s=$(date +%s)
  if bash -c "$cmd" >> "logs/${GRID:-grid}_worker${WORKER}_runs.log" 2>&1; then st=OK; else st=FAIL; fi
  echo "$(date '+%F %T') [$i] $st in $(( ($(date +%s) - s) / 60 )) min"
done < "$LIST"
echo "$(date '+%F %T') worker $WORKER finished"
