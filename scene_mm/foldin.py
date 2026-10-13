"""Embedding NEW cells (e.g. a new donor) into a trained model without retraining it.

* aggregate(): zero-training baseline. A cell is the weighted mean of the positions of its features,
  weights = count / (mean count of that feature in the reference), so ubiquitous genes do not dominate.
  With proteins, the RNA and protein averages are combined with equal weight.
* fold_in(): all feature-side parameters of the reference model are frozen (positions, feature effects,
  dispersions, background, alpha, known batch rows). Only the new cells' positions and cell effects
  are optimised with the same likelihood, plus one new batch row per unseen batch (e.g. the new donor).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from . import data as D
from .model import ModalitySpec, MultiSCENE


def _ref_means(mods_ref):
    """Mean count per feature in the reference (training) data."""
    out = {}
    for name, m in mods_ref.items():
        X = m.X_train
        out[name] = np.asarray(X.mean(axis=0)).ravel() + 1e-3
    return out


def aggregate(model, mods_q, ref_means, weighting="mean_normalized"):
    """Zero-training embedding of query cells: weighted average of feature positions."""
    parts = []
    for name, m in mods_q.items():
        W = model.mods[name].W.detach().cpu().numpy()
        X = m.X_train
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        A = X / ref_means[name][None, :] if weighting == "mean_normalized" else X
        A = A / (A.sum(axis=1, keepdims=True) + 1e-12)
        parts.append(A @ W)
    return np.mean(parts, axis=0).astype(np.float32)


def _query_model(ref: MultiSCENE, n_query, n_new_batches, device):
    """Copy the reference feature side into a model with n_query cells; add new batch rows."""
    specs = []
    for name, m in ref.mods.items():
        s = copy.copy(m.spec)
        s.batch_levels = [(U.shape[0] + n_new, kind) for (U, kind, n_new)
                          in zip(m.batch.U, m.batch.kinds, n_new_batches)]
        specs.append(s)
    q = MultiSCENE(n_query, ref.latent_dim, specs).to(device)
    for name, m in ref.mods.items():
        qm = q.mods[name]
        for pname in ("W", "feat_re", "feat_re_pi", "raw_alpha", "raw_theta", "bg"):
            if hasattr(m, pname):
                getattr(qm, pname).data.copy_(getattr(m, pname).data)
                getattr(qm, pname).requires_grad_(False)
        for lvl, (U, V) in enumerate(zip(m.batch.U, m.batch.V)):
            qU = qm.batch.U[lvl]
            qU.data[: U.shape[0]] = U.data
            if V.numel():
                qm.batch.V[lvl].data.copy_(V.data)
                qm.batch.V[lvl].requires_grad_(False)
            n_known = U.shape[0]
            if n_new_batches[lvl] == 0:
                qU.requires_grad_(False)
            else:   # only the new rows are trainable
                qU.register_hook(lambda g, k=n_known: torch.cat([torch.zeros_like(g[:k]), g[k:]]))
    return q


def fold_in(ref: MultiSCENE, mods_q, batch_ids_q, n_new_batches, z_init=None, epochs=200, lr=0.05,
            block_size=2048, device="cpu", seed=0, log=print):
    torch.manual_seed(seed)
    n = next(iter(mods_q.values())).X_train.shape[0]
    q = _query_model(ref, n, n_new_batches, device)
    if z_init is not None:
        q.Z.data.copy_(torch.as_tensor(z_init, device=device))
    trainable = [p for p in q.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": [q.Z], "weight_decay": 1e-3},
                             {"params": [p for p in trainable if p is not q.Z], "weight_decay": 0.0}], lr=lr)
    n_entries = sum(n * m.n_features for m in mods_q.values())
    for m in mods_q.values():
        m.cache_on(device)
    blocks = [np.arange(s, min(s + block_size, n)) for s in range(0, n, block_size)]
    t0 = time.time()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for cells in blocks:
            ct = torch.as_tensor(cells, device=device)
            xs, ms = {}, {}
            for name, m in mods_q.items():
                xs[name], ms[name] = m.block(cells, device)
            bid = {k: [i[ct] for i in batch_ids_q] if batch_ids_q else None for k in mods_q}
            ll, _ = q.block_log_lik(ct, xs, bid, ms)
            (-ll / n_entries).backward()
            total += float(ll.detach())
        opt.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            log(f"  fold-in epoch {epoch:4d}  loss {-total / n_entries:.4f}  ({time.time() - t0:.0f}s)")
    return q
