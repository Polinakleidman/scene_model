"""Prepare the two CITE-seq datasets for multimodal SCENE (RNA counts in layers['counts'],
protein (ADT) counts in obsm['protein_counts'] as a DataFrame).

  pbmc    : the 30k-cell PBMC subset from the reproduction (same cells as the SCENE reproduction runs)
  neurips : NeurIPS 2021 BMMC CITE-seq (GSE194122), random 30k cells (seed 0), GEX + 134 ADT
Usage: python scripts/prepare_data.py [pbmc|neurips|all]
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJ = Path(__file__).resolve().parents[1]
REPRO = PROJ / "reproduction" / "SCENE-reproducibility" / "data"
OUT = PROJ / "data"


def pbmc():
    src = REPRO / "pbmc_cite_seq_sub30k.h5ad"
    a = ad.read_h5ad(src)
    keep = ["donor", "time", "celltype.l1", "celltype.l2", "celltype.l3", "lane"]
    b = ad.AnnData(X=sp.csr_matrix(a.layers["counts"]), obs=a.obs[keep].copy(), var=a.var[[]].copy())
    b.layers["counts"] = b.X.copy()
    b.obsm["protein_counts"] = a.obsm["protein_counts"].astype(np.float32)
    b.write_h5ad(OUT / "pbmc_mm_sub30k.h5ad")
    print("pbmc", b.shape, "proteins", b.obsm["protein_counts"].shape[1])


def neurips(n=30000, seed=0):
    src = REPRO / "raw" / "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad"
    a = ad.read_h5ad(src)
    idx = np.sort(np.random.default_rng(seed).choice(a.n_obs, n, replace=False))
    a = a[idx]
    gex = (a.var["feature_types"] == "GEX").to_numpy()
    adt = ~gex
    counts = sp.csr_matrix(a.layers["counts"])
    obs = a.obs[["cell_type", "batch", "Site", "DonorNumber"]].copy()
    b = ad.AnnData(X=counts[:, gex].astype(np.float32), obs=obs, var=pd.DataFrame(index=a.var_names[gex]))
    b.layers["counts"] = b.X.copy()
    P = counts[:, adt].toarray().astype(np.float32)
    b.obsm["protein_counts"] = pd.DataFrame(P, index=b.obs_names, columns=a.var_names[adt])
    b.write_h5ad(OUT / "neurips_mm_sub30k.h5ad")
    print("neurips", b.shape, "proteins", P.shape[1])


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(exist_ok=True)
    if what in ("pbmc", "all"):
        pbmc()
    if what in ("neurips", "all"):
        neurips()
