"""Prepare PBMC CITE-seq like `prepare_data.py pbmc`, but make it writable with current anndata.

Newer anndata/h5py refuse '/' in keys, and one protein column is named 'CD66a/c/e'.
We replace '/' with '_' in the protein_counts column names; the RNA counts used by SCENE are unchanged.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent / "SCENE-reproducibility"
sys.path.insert(0, str(REPO / "scripts"))

import scvi  # noqa: E402
from prepare_data import save  # noqa: E402

raw, out = REPO / "data" / "raw", REPO / "data" / "pbmc_cite_seq.h5ad"
a = scvi.data.pbmc_seurat_v4_cite_seq(save_path=str(raw))
for k in list(a.obsm.keys()):
    if hasattr(a.obsm[k], "columns"):
        a.obsm[k].columns = [str(c).replace("/", "_") for c in a.obsm[k].columns]
out.unlink(missing_ok=True)
save(a, out, "scvi.data.pbmc_seurat_v4_cite_seq; GSE164378 ('/' -> '_' in protein names)")
