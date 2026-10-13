"""Fast unit tests for multimodal SCENE (CPU, synthetic data). Run: python -m pytest -q tests"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene_mm import Config, fit  # noqa: E402
from scene_mm import data as D  # noqa: E402
from scene_mm.evaluate import protein_gene_pairs  # noqa: E402
from scene_mm.foldin import _ref_means, aggregate, fold_in  # noqa: E402
from scene_mm.model import Modality, ModalitySpec  # noqa: E402


def toy_adata(n=120, g=30, p=6, seed=0):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, 3, n)
    rate = rng.gamma(0.5, 1.0, (3, g))[groups] * 2
    X = rng.poisson(rate * (rng.random((n, g)) > 0.5)).astype(np.float32)
    P = rng.poisson(rng.gamma(2.0, 5.0, (3, p))[groups] + 3).astype(np.float32)
    a = ad.AnnData(X=sp.csr_matrix(X), obs=pd.DataFrame({"donor": rng.choice(["A", "B", "C"], n),
                                                         "label": groups.astype(str)}, index=[f"c{i}" for i in range(n)]),
                   var=pd.DataFrame(index=[f"G{j}" for j in range(g)]))
    a.layers["counts"] = a.X.copy()
    a.obsm["protein_counts"] = pd.DataFrame(P, index=a.obs_names, columns=[f"CD{k}" for k in range(p)])
    return a


@pytest.mark.parametrize("lik", ["zip", "zip_decoupled", "poisson", "nb"])
def test_fast_loglik_equals_dense(lik):
    torch.manual_seed(0)
    x = torch.poisson(torch.rand(40, 25) * 3) * (torch.rand(40, 25) > 0.5)
    m = Modality(ModalitySpec("rna", 25, lik), 3, 40)
    p = m.params(torch.randn(40, 3), torch.arange(40), None)
    assert torch.allclose(m.log_prob(p, x).sum(), m.log_lik_sum(p, x), rtol=1e-4)


def test_protein_nb_bg_mean_includes_background():
    m = Modality(ModalitySpec("protein", 5, "nb_bg"), 2, 10)
    p = m.params(torch.randn(10, 2) * 50, torch.arange(10), None)   # cells far from all proteins
    assert torch.allclose(p["mu"], p["bg"], rtol=1e-3)                 # -> only background remains


def test_mask_rna_hides_nonzeros_and_samples_true_zeros():
    X = sp.random(50, 40, density=0.3, format="csr", random_state=0, dtype=np.float32)
    X.data = np.ceil(X.data * 5)
    X_tr, h = D.mask_rna(X, 0.1, seed=1)
    assert X_tr.nnz == X.nnz - len(h["values"])
    assert np.all(np.asarray(X[h["neg_rows"], h["neg_cols"]]).ravel() == 0)
    assert np.all(np.asarray(X_tr[h["rows"], h["cols"]]).ravel() == 0)


def test_fit_runs_and_loss_decreases():
    a = toy_adata()
    cfg = Config(latent_dim=3, epochs=60, use_protein=True, rna_likelihood="nb", eval_every=59, block_size=50)
    model, mods, _, hist, m = fit(a, cfg, log=lambda s: None)
    assert hist["loss"][-1] < hist["loss"][0]
    assert 0.5 < m["rna"]["auc_roc"] <= 1 and "protein" in m


def test_foldin_keeps_reference_features_frozen():
    a = toy_adata()
    ref, qry = a[a.obs.donor != "C"].copy(), a[a.obs.donor == "C"].copy()
    cfg = Config(latent_dim=3, epochs=30, use_protein=True, batch_keys=["donor"], val_frac=0.0, block_size=50)
    model, mods, (ids, cats), _, _ = fit(ref, cfg, log=lambda s: None)
    mods_q = D.build(qry, True, 0.0, 0)
    ids_q, cats_q = D.batch_ids(qry, ["donor"], "cpu", categories=cats)
    assert cats_q[0][:2] == cats[0] and cats_q[0][-1] == "C"
    z0 = aggregate(model, mods_q, _ref_means(mods))
    q = fold_in(model, mods_q, ids_q, [1], z_init=z0, epochs=20, log=lambda s: None)
    for name in ("rna", "protein"):
        assert torch.equal(q.mods[name].W, model.mods[name].W)
        assert torch.equal(q.mods[name].feat_re, model.mods[name].feat_re)
        U_ref, U_q = model.mods[name].batch.U[0], q.mods[name].batch.U[0]
        assert torch.equal(U_q[: U_ref.shape[0]], U_ref)                    # known donors unchanged
    assert not torch.equal(q.Z, torch.as_tensor(z0))                            # new cells did move


def test_protein_gene_mapping_strips_clone_suffix():
    pairs = protein_gene_pairs(["CD8", "CD3-1", "Rat-IgG1-1", "CD20"], ["CD8A", "CD3E", "MS4A1", "ACTB"])
    assert [(p, g) for _, _, p, g in pairs] == [("CD8", "CD8A"), ("CD3-1", "CD3E"), ("CD20", "MS4A1")]
