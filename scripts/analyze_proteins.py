"""Which proteins land next to their own gene, and why?

For the multimodal PBMC and NeurIPS models (RNA+protein, NB, 16D, seeds 0-2) this computes, per protein:
  * foreground share: mean over cells of fg / (fg + bg), i.e. how much of the predicted ADT count the model
    attributes to cell-specific signal rather than to the ambient background;
  * the mean expression of its gene (RNA);
and joins them with the percentile rank of the own gene (from results_tables/protein_pairs.csv).
Writes results_tables/protein_signal.csv and prints Spearman correlations.
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from scene_mm.train import load_run  # noqa: E402

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"


def main():
    pairs = pd.read_csv(PROJ / "results_tables/protein_pairs.csv")
    rows = []
    for ds in ["pbmc", "neurips"]:
        adata = ad.read_h5ad(PROJ / f"data/{ds}_mm_sub30k.h5ad")
        for seed in [0, 1, 2]:
            model, mods, ids, cfg = load_run(PROJ / f"results/{ds}_full_rna+prot_nb_d16_s{seed}", adata, DEVICE)
            pm = model.mods["protein"]
            with torch.no_grad():
                fg_share = torch.zeros(pm.W.shape[0], device=DEVICE)
                n = adata.n_obs
                for s in range(0, n, 4096):
                    cells = torch.arange(s, min(s + 4096, n), device=DEVICE)
                    bid = [i[cells] for i in ids] if ids else None
                    p = pm.params(model.Z[cells], cells, bid)
                    fg_share += (p["fg"] / p["mu"]).sum(0)
                fg_share = (fg_share / n).cpu().numpy()
            gene_mean = pd.Series(np.asarray(mods["rna"].X_train.mean(axis=0)).ravel(), index=mods["rna"].feature_names)
            prot_mean = mods["protein"].X_train.mean(axis=0)
            sub = pairs[(pairs.dataset == ds) & (pairs.modality == "rna+prot") & (pairs.likelihood == "nb")
                        & (pairs.dim == 16) & (pairs.seed == seed)]
            names = list(mods["protein"].feature_names)
            for _, r in sub.iterrows():
                k = names.index(r.protein)
                rows.append({"dataset": ds, "seed": seed, "protein": r.protein, "gene": r.gene, "rank_pct": r.rank_pct,
                             "fg_share": float(fg_share[k]), "protein_mean": float(prot_mean[k]),
                             "gene_mean": float(gene_mean[r.gene])})
    df = pd.DataFrame(rows)
    df.to_csv(PROJ / "results_tables/protein_signal.csv", index=False)
    for ds, d in df.groupby("dataset"):
        a = d.groupby(["protein", "gene"]).mean(numeric_only=True)
        print(f"{ds}: {len(a)} pairs")
        for col in ["fg_share", "gene_mean", "protein_mean"]:
            rho, p = spearmanr(a[col], a["rank_pct"])
            print(f"   Spearman(rank of own gene, {col}) = {rho:+.2f}  (p = {p:.1e})")


if __name__ == "__main__":
    main()
