"""Baselines for the multimodal experiments: scVI (RNA) and totalVI (RNA + protein), scvi-tools.

  full : train on the whole dataset -> clustering metrics of the latent space.
  lodo : train on all donors but one; map the held-out donor with scArches
         (load_query_data + query training, reference weights frozen) -> kNN label transfer.
Writes <out>/<run_id>/metrics.json (+ latent .npy).

  python scripts/run_baseline.py --dataset pbmc --method totalvi --mode lodo --lodo-donor P3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))
from run_experiment import DATASETS  # noqa: E402
from scene_mm.evaluate import cluster_metrics, label_transfer  # noqa: E402


def setup(cls, adata, method, batch_key):
    kw = {"layer": "counts", "batch_key": batch_key}
    if method == "totalvi":
        kw["protein_expression_obsm_key"] = "protein_counts"
    cls.setup_anndata(adata, **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--method", choices=["scvi", "totalvi"], required=True)
    p.add_argument("--mode", choices=["full", "lodo"], default="full")
    p.add_argument("--lodo-donor", default=None)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--query-epochs", type=int, default=200)
    p.add_argument("--accelerator", default="cpu")
    p.add_argument("--n-cells", type=int, default=None)
    p.add_argument("--out", default=str(PROJ / "results"))
    a = p.parse_args()

    import scvi
    scvi.settings.seed = a.seed
    cls = scvi.model.TOTALVI if a.method == "totalvi" else scvi.model.SCVI
    spec = DATASETS[a.dataset]
    rid = f"{a.dataset}_{a.mode}_{a.method}_d{a.dim}_s{a.seed}" + (f"_{a.lodo_donor}" if a.mode == "lodo" else "")
    out = Path(a.out) / rid
    if (out / "metrics.json").exists():
        print(f"exists, skipping: {out}")
        return
    adata = ad.read_h5ad(spec["path"])
    if a.n_cells:
        adata = adata[np.sort(np.random.default_rng(0).choice(adata.n_obs, a.n_cells, replace=False))].copy()
    batch_key = (spec["batch_keys"] or [spec["donor"]])[0]
    train_kw = {"max_epochs": a.epochs, "accelerator": a.accelerator}
    t0 = time.time()
    res = {"run_id": rid, "dataset": a.dataset, "method": a.method, "mode": a.mode}
    out.mkdir(parents=True, exist_ok=True)

    if a.mode == "full":
        setup(cls, adata, a.method, batch_key)
        m = cls(adata, n_latent=a.dim)
        m.train(**train_kw)
        Z = m.get_latent_representation()
        np.save(out / "latent.npy", Z)
        res["clustering"] = {lab: cluster_metrics(Z, adata.obs[lab].astype(str).to_numpy(), a.seed)
                             for lab in spec["labels"]}
    else:
        is_q = adata.obs[spec["donor"]].astype(str).to_numpy() == a.lodo_donor
        ref, qry = adata[~is_q].copy(), adata[is_q].copy()
        setup(cls, ref, a.method, batch_key)
        m = cls(ref, n_latent=a.dim)
        m.train(**train_kw)
        Z_ref = m.get_latent_representation()
        cls.prepare_query_anndata(qry, m)
        q = cls.load_query_data(qry, m)
        t1 = time.time()
        q.train(max_epochs=a.query_epochs, accelerator=a.accelerator, plan_kwargs={"weight_decay": 0.0})
        res["query_time_s"] = round(time.time() - t1, 1)
        Z_q = q.get_latent_representation()
        np.save(out / "latent_ref.npy", Z_ref)
        np.save(out / "latent_query.npy", Z_q)
        res["transfer"] = {lab: {"scarches": label_transfer(Z_ref, ref.obs[lab].astype(str).to_numpy(),
                                                            Z_q, qry.obs[lab].astype(str).to_numpy())}
                           for lab in spec["labels"]}
    res["total_time_s"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str)[:2000])


if __name__ == "__main__":
    main()
