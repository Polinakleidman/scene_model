#!/usr/bin/env bash
# Set up SIMBA and TCR on the cluster login node (needs internet, no root):
#   1. micromamba (user-level conda) in ~/bin
#   2. envs/simba  from environment_simba.yaml   (python 3.10, torch 2.3, dgl, PyTorch-BigGraph)
#   3. envs/data_r from environment_data_r.yaml  (R 4.4 + Seurat 5, only for TCR data conversion)
#   4. TCR data: download the two GEO Seurat objects, convert (R), then prepare_data.py tcr
# Safe to re-run: finished steps are skipped. Log: logs/setup_simba_tcr.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$ROOT/SCENE-reproducibility"
MM="$HOME/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
mkdir -p "$ROOT/envs" "$ROOT/logs" "$REPO/data/raw"

if [ ! -x "$MM" ]; then
  echo "== micromamba"
  mkdir -p "$HOME/bin"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME" bin/micromamba
fi
"$MM" --version

# Check by import, not by file: an interrupted create leaves python without the pip packages.
if ! "$ROOT/envs/simba/bin/python" -c "import simba, torchbiggraph" 2>/dev/null; then
  echo "== env simba $(date +%T)"
  rm -rf "$ROOT/envs/simba"
  # Same versions as environment_simba.yaml, installed in two steps: dgl 2.2.0 is not on PyPI
  # (it comes from data.dgl.ai), and SIMBA runs on CPU, so torch 2.3.0 is the CPU build.
  "$MM" create -y -p "$ROOT/envs/simba" -c conda-forge python=3.10 pip git ipykernel
  SP="$ROOT/envs/simba/bin/pip"
  "$SP" install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
  sed -n '/- pip:/,$p' "$REPO/environment_simba.yaml" | sed -n 's/^ *- //p' | grep -v '^pip:' \
    > "$ROOT/envs/simba_requirements.txt"
  "$SP" install -r "$ROOT/envs/simba_requirements.txt" -f https://data.dgl.ai/wheels/torch-2.3/repo.html \
    --extra-index-url https://download.pytorch.org/whl/cpu
fi
"$ROOT/envs/simba/bin/python" -c "import simba, torchbiggraph, torch; print('simba ok, torch', torch.__version__)"

if ! "$ROOT/envs/data_r/bin/Rscript" -e "library(Seurat); library(zellkonverter)" >/dev/null 2>&1; then
  echo "== env data_r $(date +%T)"
  rm -rf "$ROOT/envs/data_r"
  "$MM" create -y -p "$ROOT/envs/data_r" -f "$REPO/environment_data_r.yaml"
fi

cd "$REPO"
GEO=https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4450nnn
for s in GSM4450386_stimulated GSM4450387_unstimulated; do
  f="data/raw/${s}_full_seurat.rds"
  if [ ! -f "$f" ]; then
    echo "== download $s $(date +%T)"
    curl -L --fail --retry 5 -C - -o "$f.gz" "$GEO/${s%%_*}/suppl/${s}_full_seurat.rds.gz"
    gunzip -f "$f.gz"
  fi
done
if [ ! -f data/raw/GSM4450387_unstimulated.h5ad ]; then
  echo "== R conversion $(date +%T)"
  # authors' prepare_tcr.R fails with current SeuratObject (see prepare_tcr_fix.R)
  "$ROOT/envs/data_r/bin/Rscript" "$ROOT/prepare_tcr_fix.R" data/raw
  "$ROOT/.venv/bin/python" "$ROOT/prepare_tcr_fix.py" data/raw
fi
if [ ! -f data/tcr_stim_data.h5ad ]; then
  "$ROOT/.venv/bin/python" scripts/prepare_data.py tcr
fi
echo "DONE $(date +%T)"
ls -la data/tcr_stim_data.h5ad
