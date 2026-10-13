#!/bin/bash
# Toy run of the reproduction pipeline (~1 min on a laptop CPU, no downloads).
# Uses the small cortex dataset bundled in the authors' repository (3,005 cells x 19,972 genes):
# trains SCENE (16D, ZIP) and the PCA baseline with the authors' scripts, then prints KMeans NMI/ARI.
# Run from reproduction/ after cloning the authors' code into SCENE-reproducibility/ (see README).
set -euo pipefail
cd "$(dirname "$0")/SCENE-reproducibility"
PY=${PYTHON:-python}
EPOCHS=${EPOCHS:-200}

$PY scripts/train_latent.py --input_h5ad data/cortex.h5ad --results_dir results/toy --run_name SCENE_toy \
    --latent_dim 16 --epochs "$EPOCHS" --device cpu --lr 0.05 --layer counts --out_emb scLDM \
    --cell_init random --gene_init random --seed 0 --rank 2 --loss_type ZIP --split_seed 0 --no_validate --pi_only
$PY scripts/train_pca_harmony.py --input_h5ad data/cortex.h5ad --results_dir results/toy --run_name PCA_toy \
    --seed 0 --embedding_dim 50

$PY - <<'EOF'
import anndata as ad, numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
y = ad.read_h5ad("data/cortex.h5ad", backed="r").obs["cell_type"].astype(str).to_numpy()
k = len(set(y))
for name, path in [("SCENE (16D)", "results/toy/SCENE_toy/scLDM_cell_latent.npy"),
                   ("PCA (50D)", "results/toy/PCA_toy/X_pca_vanilla_cell_latent.npy")]:
    Z = np.load(path)
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
    print(f"{name:12s}  KMeans NMI = {normalized_mutual_info_score(y, lab):.3f}   ARI = {adjusted_rand_score(y, lab):.3f}")
print("Paper / our full run (Fig. 4, cortex): SCENE is the best method on this dataset.")
EOF
