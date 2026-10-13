"""Why does NB have a better held-out log-likelihood but a worse Poisson deviance than ZIP?

Loads the trained RNA-only models (PBMC, 16D, seed 0), recomputes predictions at exactly the held-out
entries used in training, and breaks the metrics down by the observed count.
Writes results_tables/likelihood_calibration.csv and likelihood_theta.csv.
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from scene_mm.train import _params_at, load_run  # noqa: E402

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
BINS = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 10**9)]


def main():
    adata = ad.read_h5ad(PROJ / "data/pbmc_mm_sub30k.h5ad")
    rows, thetas = [], []
    for lik in ["zip", "nb"]:
        run = PROJ / f"results/pbmc_full_rna_{lik}_d16_s0"
        model, mods, ids, cfg = load_run(run, adata, DEVICE)
        h = mods["rna"].heldout
        mod = model.mods["rna"]
        with torch.no_grad():
            pp = _params_at(model, "rna", h["rows"], h["cols"], ids, cfg)
            pn = _params_at(model, "rna", h["neg_rows"], h["neg_cols"], ids, cfg)
            x = torch.as_tensor(h["values"])
            mu_pos = mod.mean_given_positive(pp).clamp_min(1e-8)
            dev = (2 * (x * torch.log(x / mu_pos) - (x - mu_pos))).numpy()
            nll_pos = (-mod.log_prob(pp, x)).numpy()
            nll_neg = (-mod.log_prob(pn, torch.zeros(len(h["neg_rows"])))).numpy()
            p_det_pos = mod.prob_detect(pp).numpy()
            p_det_neg = mod.prob_detect(pn).numpy()
        xv = h["values"]
        rows.append({"likelihood": lik, "bin": "zeros (true negatives)", "n": len(nll_neg), "mean_obs": 0.0,
                     "mean_pred_given_pos": np.nan, "mean_deviance": np.nan,
                     "mean_nll": float(nll_neg.mean()), "mean_p_detect": float(p_det_neg.mean())})
        for lo, hi in BINS:
            m = (xv >= lo) & (xv <= hi)
            rows.append({"likelihood": lik, "bin": f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"),
                         "n": int(m.sum()), "mean_obs": float(xv[m].mean()),
                         "mean_pred_given_pos": float(mu_pos.numpy()[m].mean()),
                         "median_pred_given_pos": float(np.median(mu_pos.numpy()[m])),
                         "mean_deviance": float(dev[m].mean()), "share_of_total_deviance": float(dev[m].sum() / dev.sum()),
                         "mean_nll": float(nll_pos[m].mean()), "mean_p_detect": float(p_det_pos[m].mean())})
        if lik == "nb":
            th = mod.theta.detach().cpu().numpy()
            mean_expr = np.asarray(mods["rna"].X_train.mean(axis=0)).ravel()
            thetas = pd.DataFrame({"gene": mods["rna"].feature_names, "theta": th, "mean_expr": mean_expr})
    out = PROJ / "results_tables"
    df = pd.DataFrame(rows)
    df.to_csv(out / "likelihood_calibration.csv", index=False)
    thetas.to_csv(out / "likelihood_theta.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print("\nNB theta: median %.2f; share of genes with theta < 1: %.2f" % (thetas.theta.median(), (thetas.theta < 1).mean()))


if __name__ == "__main__":
    main()
