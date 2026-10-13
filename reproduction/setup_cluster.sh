#!/usr/bin/env bash
# Create the Python environment on a Linux GPU node (the Mac .venv is not portable).
# Needs Python >= 3.10 (e.g. `module load python/3.11` or a conda env first).
# Can run on the login node (pip needs internet; compute nodes may not have it).
# The CUDA check below prints "cuda: False" there, which is fine.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
PYTHON="${PYTHON:-python3}"
if [ ! -x .venv/bin/pip ]; then
  rm -rf .venv
  # Ubuntu without python3-venv has no ensurepip (and no sudo): bootstrap pip with get-pip.py.
  "$PYTHON" -m venv .venv 2>/dev/null || {
    rm -rf .venv
    "$PYTHON" -m venv --without-pip .venv
    curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
  }
fi
.venv/bin/pip install --upgrade pip
# Default Linux torch wheels include CUDA.
.venv/bin/pip install torch pytorch-lightning torchmetrics joblib numpy pandas scipy scikit-learn \
  anndata scanpy h5py statsmodels tqdm leidenalg python-igraph umap-learn harmonypy \
  ipykernel nbformat matplotlib adjusttext seaborn fastcluster nbclient scvi-tools pytest
.venv/bin/pip install -e SCENE-reproducibility/scene
# The default torch wheel needs a newer driver than the cluster's (CUDA 12.8 driver on the
# MBZUAI gpu nodes): reinstall torch built for CUDA 12.8. Override with TORCH_INDEX.
.venv/bin/pip install --force-reinstall torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
.venv/bin/python -c "import torch, scene; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
echo
echo "Next (from this folder):"
echo "  sbatch run_cluster.sbatch        # full reproduction, chains 8 h jobs until done"
echo "  squeue -u \$USER                  # job status"
echo "  tail -f logs/queue.log            # progress"
