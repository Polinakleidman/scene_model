"""Build the small CITE-seq toy dataset shipped in toy_data/ (for quick tests without downloads).

Source: the 30k-cell PBMC CITE-seq subset (data/pbmc_mm_sub30k.h5ad, see scripts/prepare_data.py).
Toy: 250 random cells per donor (8 donors, 2,000 cells), the 1,500 genes detected in most cells plus every gene
that codes one of the measured proteins, and all 217 proteins. Raw counts in X, proteins in obsm['protein_counts'].
Run from scene_project/:  python scripts/make_toy_data.py
"""
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

PROJ = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJ))
from scene_mm.evaluate import protein_gene_pairs  # noqa: E402

src = ad.read_h5ad(PROJ / "data/pbmc_mm_sub30k.h5ad")
rng = np.random.default_rng(0)
cells = np.sort(np.concatenate([rng.choice(np.where(src.obs["donor"].astype(str) == d)[0], 250, replace=False)
                                for d in sorted(src.obs["donor"].astype(str).unique())]))
X = sp.csr_matrix(src.layers["counts"][cells])
detected = np.asarray((X > 0).sum(0)).ravel()
top = set(np.argsort(-detected)[:1500])
names = list(src.var_names)
paired = {gi for _, gi, _, _ in protein_gene_pairs(list(src.obsm["protein_counts"].columns), names)}
genes = np.array(sorted(top | paired))

toy = ad.AnnData(X=X[:, genes].astype(np.int32), obs=src.obs.iloc[cells].copy(), var=src.var.iloc[genes].copy())  # raw counts in X
toy.obsm["protein_counts"] = src.obsm["protein_counts"].iloc[cells].copy()
out = PROJ / "toy_data" / "toy_pbmc_cite.h5ad"
out.parent.mkdir(exist_ok=True)
toy.write_h5ad(out, compression="gzip")
print(f"wrote {out.relative_to(PROJ)}: {toy.n_obs} cells x {toy.n_vars} genes, "
      f"{toy.obsm['protein_counts'].shape[1]} proteins ({len(paired)} protein-coding genes kept), "
      f"{out.stat().st_size / 1e6:.1f} MB")
