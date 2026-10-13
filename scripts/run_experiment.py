"""One multimodal-SCENE experiment. Writes <out>/<run_id>/{metrics.json, embedding.h5ad, model.pt}.

Modes
  full : train on the whole dataset (10% held out) -> held-out likelihood metrics, clustering of cell
         types, protein<->gene alignment (if proteins are used).
  lodo : leave one donor out. Train on the other donors, then embed the held-out donor by
         (a) aggregation (no training) and (b) fold-in (frozen features), and transfer cell-type labels
         from the reference by kNN.

Example
  python scripts/run_experiment.py --dataset pbmc --mode full --rna-lik nb --protein --dim 16 --seed 0
  python scripts/run_experiment.py --dataset pbmc --mode lodo --lodo-donor P3 --protein --dim 16
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
from scene_mm import Config, export, fit  # noqa: E402
from scene_mm import data as D  # noqa: E402
from scene_mm.evaluate import cluster_metrics, label_transfer, protein_gene_alignment  # noqa: E402
from scene_mm.foldin import _ref_means, aggregate, fold_in  # noqa: E402

DATASETS = {
    "pbmc": {"path": PROJ / "data/pbmc_mm_sub30k.h5ad", "labels": ["celltype.l1", "celltype.l2"],
             "donor": "donor", "batch_keys": []},
    "neurips": {"path": PROJ / "data/neurips_mm_sub30k.h5ad", "labels": ["cell_type"],
                "donor": "DonorNumber", "batch_keys": ["batch"]},
}


def run_id(a):
    mod = "rna+prot" if a.protein else "rna"
    base = f"{a.dataset}_{a.mode}_{mod}_{a.rna_lik}_d{a.dim}_s{a.seed}"
    return base + (f"_{a.lodo_donor}" if a.mode == "lodo" else "") + (a.tag and f"_{a.tag}" or "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--mode", choices=["full", "lodo"], default="full")
    p.add_argument("--rna-lik", default="zip", choices=["zip", "nb", "zip_decoupled", "poisson"])
    p.add_argument("--protein", action="store_true")
    p.add_argument("--protein-weight", type=float, default=1.0)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--foldin-epochs", type=int, default=200)
    p.add_argument("--lodo-donor", default=None)
    p.add_argument("--n-cells", type=int, default=None, help="random subset (smoke tests)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--block-size", type=int, default=4096)
    p.add_argument("--out", default=str(PROJ / "results"))
    p.add_argument("--tag", default="")
    a = p.parse_args()
    spec = DATASETS[a.dataset]
    out = Path(a.out) / run_id(a)
    if (out / "metrics.json").exists():
        print(f"exists, skipping: {out}")
        return
    adata = ad.read_h5ad(spec["path"])
    if a.n_cells:
        adata = adata[np.sort(np.random.default_rng(0).choice(adata.n_obs, a.n_cells, replace=False))].copy()
    # donors are a batch level in LODO so that fold-in can learn the new donor's offset
    batch_keys = list(spec["batch_keys"]) or ([spec["donor"]] if a.mode == "lodo" else [])
    cfg = Config(latent_dim=a.dim, epochs=a.epochs, rna_likelihood=a.rna_lik, use_protein=a.protein,
                 protein_weight=a.protein_weight, batch_keys=batch_keys, seed=a.seed, device=a.device,
                 block_size=a.block_size)
    t0 = time.time()
    result = {"run_id": out.name, "dataset": a.dataset, "mode": a.mode}

    if a.mode == "full":
        model, mods, (ids, cats), hist, held = fit(adata, cfg)
        Z = model.Z.detach().cpu().numpy()
        result["heldout"] = held
        result["clustering"] = {lab: cluster_metrics(Z, adata.obs[lab].astype(str).to_numpy(), a.seed)
                                for lab in spec["labels"]}
        if a.protein:
            result["protein_gene"] = protein_gene_alignment(
                model.mods["protein"].W.detach().cpu().numpy(), model.mods["rna"].W.detach().cpu().numpy(),
                mods["protein"].feature_names, mods["rna"].feature_names)
        export(model, mods, cats, adata, cfg, hist, held, out)

    else:
        donor = a.lodo_donor
        is_q = adata.obs[spec["donor"]].astype(str).to_numpy() == donor
        ref, qry = adata[~is_q].copy(), adata[is_q].copy()
        model, mods, (ids, cats), hist, held = fit(ref, cfg)
        result["heldout_ref"] = held
        Z_ref = model.Z.detach().cpu().numpy()
        mods_q = D.build(qry, a.protein, 0.0, a.seed)
        ids_q, cats_q = D.batch_ids(qry, batch_keys, a.device, categories=cats)
        n_new = [len(cq) - len(c) for cq, c in zip(cats_q, cats)]
        Z_agg = aggregate(model, mods_q, _ref_means(mods))
        t1 = time.time()
        q = fold_in(model, mods_q, ids_q, n_new, z_init=Z_agg, epochs=a.foldin_epochs, device=a.device, seed=a.seed)
        result["foldin_time_s"] = round(time.time() - t1, 1)
        Z_fold = q.Z.detach().cpu().numpy()
        result["transfer"] = {}
        for lab in spec["labels"]:
            y_ref, y_q = ref.obs[lab].astype(str).to_numpy(), qry.obs[lab].astype(str).to_numpy()
            result["transfer"][lab] = {"aggregate": label_transfer(Z_ref, y_ref, Z_agg, y_q),
                                       "foldin": label_transfer(Z_ref, y_ref, Z_fold, y_q)}
        out.mkdir(parents=True, exist_ok=True)
        emb = ad.AnnData(obs=qry.obs.copy())
        emb.obsm["X_aggregate"], emb.obsm["X_foldin"] = Z_agg, Z_fold
        emb.write_h5ad(out / "query_embedding.h5ad")
        export(model, mods, cats, ref, cfg, hist, held, out)

    result["total_time_s"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k in ("heldout", "clustering", "transfer")}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
