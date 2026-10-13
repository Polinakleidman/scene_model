"""Build GSM..._{stimulated,unstimulated}.h5ad from the files written by prepare_tcr_fix.R.

Same content as the authors' zellkonverter export: X = RNA counts (cells x genes),
obs = Seurat meta.data. Then run `prepare_data.py tcr` as usual.
"""
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import scipy.io
import scipy.sparse as sp

raw = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
for sample in ["GSM4450386_stimulated", "GSM4450387_unstimulated"]:
    prefix = raw / sample
    X = sp.csr_matrix(scipy.io.mmread(f"{prefix}_counts.mtx").T, dtype="float32")
    obs = pd.read_csv(f"{prefix}_obs.csv", index_col=0)
    var = pd.DataFrame(index=Path(f"{prefix}_genes.txt").read_text().split("\n")[:-1])
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.write_h5ad(raw / f"{sample}.h5ad")
    print(sample, a.shape)
