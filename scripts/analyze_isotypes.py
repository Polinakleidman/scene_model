"""Sanity check of the protein background term with isotype controls (PBMC).

Isotype-control antibodies (Rat-IgG1 x2, Rat-IgG2b, Rat-IgG2c; the panel spells the last one "Rag-IgG2c")
do not bind any real antigen, so ALL of their counts should be background. IgM (surface IgM of B cells)
is a positive control with a strong, B-cell-specific signal.

For every protein and every PBMC multimodal model (RNA+protein, 16D, NB and ZIP, seeds 0-2):
  fg_share     : mean over cells of fg / (fg + bg), the share the model attributes to cell-specific signal
  data_spec    : in the DATA, max over cell types (celltype.l1) of mean count / overall mean
                 (1 = the same level in every cell type, i.e. no cell-type signal)
Writes results_tables/protein_fg_all.csv.
"""
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from scene_mm.train import load_run  # noqa: E402

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
ISOTYPE = re.compile(r"^(Rat|Rag|Mouse|Hamster|Armenian).*Ig", re.I)


def main():
    adata = ad.read_h5ad(PROJ / "data/pbmc_mm_sub30k.h5ad")
    P = adata.obsm["protein_counts"]
    names = list(P.columns)
    ct = adata.obs["celltype.l1"].astype(str)
    by_type = P.groupby(ct.values).mean()
    data_spec = (by_type.max() / (P.mean() + 1e-9)).reindex(names)
    rows = []
    for lik in ["nb", "zip"]:
        for seed in [0, 1, 2]:
            model, mods, ids, cfg = load_run(PROJ / f"results/pbmc_full_rna+prot_{lik}_d16_s{seed}", adata, DEVICE)
            pm = model.mods["protein"]
            fg_share = torch.zeros(len(names), device=DEVICE)
            bg_mean = torch.zeros(len(names), device=DEVICE)
            with torch.no_grad():
                for s in range(0, adata.n_obs, 4096):
                    cells = torch.arange(s, min(s + 4096, adata.n_obs), device=DEVICE)
                    p = pm.params(model.Z[cells], cells, [i[cells] for i in ids] if ids else None)
                    fg_share += (p["fg"] / p["mu"]).sum(0)
                    bg_mean += p["bg"].sum(0)
            fg_share = (fg_share / adata.n_obs).cpu().numpy()
            bg_mean = (bg_mean / adata.n_obs).cpu().numpy()
            for k, n in enumerate(names):
                rows.append({"likelihood": lik, "seed": seed, "protein": n,
                             "kind": "isotype" if ISOTYPE.match(n) else ("IgM (positive control)" if n == "IgM" else "protein"),
                             "fg_share": float(fg_share[k]), "pred_background_mean": float(bg_mean[k]),
                             "observed_mean": float(P[n].mean()), "data_spec": float(data_spec[n])})
    df = pd.DataFrame(rows)
    df.to_csv(PROJ / "results_tables/protein_fg_all.csv", index=False)
    a = df.groupby(["kind", "protein"]).mean(numeric_only=True).reset_index()
    a["fg_share_pct_rank"] = a.fg_share.rank(pct=True)
    pd.set_option("display.width", 200)
    print(a[a.kind != "protein"].round(3).to_string(index=False))
    print("\nall other proteins: fg_share median %.2f (IQR %.2f-%.2f); data_spec median %.2f"
          % (a[a.kind == "protein"].fg_share.median(), *a[a.kind == "protein"].fg_share.quantile([0.25, 0.75]),
             a[a.kind == "protein"].data_spec.median()))


if __name__ == "__main__":
    main()
