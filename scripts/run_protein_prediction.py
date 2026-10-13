"""Predict surface-protein counts of a NEW donor from its RNA only (PBMC, leave-one-donor-out).

Methods
  scene_nb / scene_zip : the multimodal LODO model (trained on the other 7 donors with RNA + protein) is reused
                         unchanged; the new donor is folded in with RNA only (proteins unseen), and its protein
                         counts are read off the geometry: mu_ik = exp(psi_i)(exp(beta_k) + exp(rho_k - ||z_i - v_k||)).
                         The unobserved protein cell effect psi_i is predicted from the RNA cell effect with a
                         linear fit on the reference cells.
  scene_nb_latentknn   : same fold-in, but proteins = mean of the 15 nearest REFERENCE cells in the SCENE latent
                         space (separates the quality of the embedding from the geometric protein read-out).
  knn                  : RNA baseline without a latent model. Log-normalise, 50 PCs on the reference, protein =
                         mean of the 15 nearest reference cells.
  totalvi              : totalVI trained on reference + new donor with the new donor's proteins marked missing
                         (the standard totalVI protein-imputation setup; it sees the query RNA during training).
Metrics per protein (isotype controls excluded): Pearson / Spearman correlation between log1p(predicted) and
log1p(observed) across the new donor's cells. Writes results_protein/<method>_<donor>/{metrics.json, per_protein.csv}.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
from scene_mm import data as D  # noqa: E402
from scene_mm.foldin import _ref_means, aggregate, fold_in  # noqa: E402
from scene_mm.train import load_run  # noqa: E402

ISOTYPE = re.compile(r"^(Rat|Rag|Mouse|Hamster|Armenian).*Ig", re.I)


def scene_predict(ref, qry, lik, donor, device, epochs):
    run = PROJ / f"results/pbmc_lodo_rna+prot_{lik}_d16_s0_{donor}"
    model, mods, ids, cfg = load_run(run, ref, device)
    mods_q = D.build(qry, False, 0.0, 0)                       # RNA only
    ids_q, cats_q = D.batch_ids(qry, cfg.batch_keys, device, categories=D.batch_ids(ref, cfg.batch_keys, "cpu")[1])
    n_new = [len(c) - len(r) for c, r in zip(cats_q, D.batch_ids(ref, cfg.batch_keys, "cpu")[1])]
    rna_means = {"rna": _ref_means(mods)["rna"]}
    z0 = aggregate(model, {"rna": mods_q["rna"]}, rna_means)
    q = fold_in(model, mods_q, ids_q, n_new, z_init=z0, epochs=epochs, device=device, log=lambda s: None)
    # protein cell effect of the new cells: linear fit psi_prot ~ psi_rna on the reference
    pr, pp = model.mods["rna"].cell_re.detach().cpu().numpy().ravel(), model.mods["protein"].cell_re.detach().cpu().numpy().ravel()
    a, b = np.polyfit(pr, pp, 1)
    psi_q = a * q.mods["rna"].cell_re.detach().cpu().numpy().ravel() + b
    pm = q.mods["protein"]
    with torch.no_grad():
        z = q.Z.detach()
        d = torch.cdist(z, pm.W)
        fg = torch.exp(pm.feat_re.T - d)
        bg = torch.exp(pm.bg.T).expand_as(fg)
        mu = torch.exp(torch.as_tensor(psi_q, device=z.device, dtype=z.dtype))[:, None] * (bg + fg)
    return mu.cpu().numpy(), model.Z.detach().cpu().numpy(), q.Z.detach().cpu().numpy()


def knn_predict(ref, qry, k=15):
    import scanpy as sc
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    def lognorm(a):
        b = ad.AnnData(X=a.layers["counts"].copy())
        sc.pp.normalize_total(b, target_sum=1e4)
        sc.pp.log1p(b)
        return b.X
    Xr, Xq = lognorm(ref), lognorm(qry)
    hv = np.argsort(-np.asarray(Xr.power(2).mean(0) - np.square(Xr.mean(0))).ravel())[:3000]
    pca = PCA(50, random_state=0).fit(Xr[:, hv].toarray())
    nn = NearestNeighbors(n_neighbors=k).fit(pca.transform(Xr[:, hv].toarray()))
    _, idx = nn.kneighbors(pca.transform(Xq[:, hv].toarray()))
    Pr = ref.obsm["protein_counts"].to_numpy()
    return Pr[idx].mean(axis=1)


def totalvi_predict(adata, is_q, accelerator, epochs):
    import scvi
    a = adata.copy()
    P = a.obsm["protein_counts"].copy()
    P.loc[is_q, :] = 0.0                     # all-zero proteins for a batch = missing in totalVI
    a.obsm["protein_counts"] = P
    scvi.model.TOTALVI.setup_anndata(a, layer="counts", batch_key="donor", protein_expression_obsm_key="protein_counts")
    m = scvi.model.TOTALVI(a, n_latent=16)
    m.train(max_epochs=epochs, accelerator=accelerator)
    # predict the query cells' proteins "as if measured" in the reference donors (the query batch has no
    # protein data), as in the totalVI imputation tutorial
    ref_batches = sorted(set(a.obs["donor"].astype(str)[~is_q]))
    _, prot = m.get_normalized_expression(indices=np.where(is_q)[0], n_samples=25, return_mean=True,
                                          include_protein_background=True, sample_protein_mixing=False,
                                          transform_batch=ref_batches)
    return np.asarray(prot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--donor", required=True)
    p.add_argument("--method", choices=["scene_nb", "scene_zip", "scene_nb_latentknn", "knn", "totalvi"], required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--accelerator", default="cpu")
    p.add_argument("--foldin-epochs", type=int, default=200)
    p.add_argument("--totalvi-epochs", type=int, default=400)
    p.add_argument("--out", default=str(PROJ / "results_protein"))
    a = p.parse_args()
    out = Path(a.out) / f"{a.method}_{a.donor}"
    if (out / "metrics.json").exists():
        print("exists, skipping", out)
        return
    adata = ad.read_h5ad(PROJ / "data/pbmc_mm_sub30k.h5ad")
    is_q = adata.obs["donor"].astype(str).to_numpy() == a.donor
    ref, qry = adata[~is_q].copy(), adata[is_q].copy()
    t0 = time.time()
    if a.method == "scene_nb_latentknn":
        from sklearn.neighbors import NearestNeighbors
        _, Zr, Zq = scene_predict(ref, qry, "nb", a.donor, a.device, a.foldin_epochs)
        _, idx = NearestNeighbors(n_neighbors=15).fit(Zr).kneighbors(Zq)
        pred = ref.obsm["protein_counts"].to_numpy()[idx].mean(axis=1)
    elif a.method.startswith("scene"):
        pred = scene_predict(ref, qry, a.method.split("_")[1], a.donor, a.device, a.foldin_epochs)[0]
    elif a.method == "knn":
        pred = knn_predict(ref, qry)
    else:
        pred = totalvi_predict(adata, is_q, a.accelerator, a.totalvi_epochs)
    obs = qry.obsm["protein_counts"].to_numpy()
    names = list(qry.obsm["protein_counts"].columns)
    rows = []
    for k, n in enumerate(names):
        if ISOTYPE.match(n):
            continue
        x, y = np.log1p(obs[:, k]), np.log1p(np.clip(pred[:, k], 0, None))
        if x.std() == 0 or y.std() == 0:
            continue
        rows.append({"protein": n, "pearson": pearsonr(x, y)[0], "spearman": spearmanr(x, y)[0]})
    pp = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    pp.to_csv(out / "per_protein.csv", index=False)
    res = {"method": a.method, "donor": a.donor, "n_proteins": len(pp), "time_s": round(time.time() - t0, 1),
           "pearson_mean": float(pp.pearson.mean()), "pearson_median": float(pp.pearson.median()),
           "spearman_mean": float(pp.spearman.mean())}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
