"""Data containers for multimodal SCENE: counts per modality, held-out entries, batch ids."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class ModalityData:
    name: str
    X_train: sp.csr_matrix | np.ndarray     # counts used for training (held-out RNA positives set to 0)
    feature_names: np.ndarray
    heldout: dict | None = None             # rows, cols, values (+ neg_rows, neg_cols for RNA)
    mask: sp.csr_matrix | None = None       # protein: True = held out (excluded from the likelihood)

    @property
    def n_features(self):
        return self.X_train.shape[1]

    _cache = None   # dense (x, mask) on the device, filled by cache_on()

    def cache_on(self, device):
        """Keep the whole dense training matrix on the device (fast; needs n_cells*n_features*4 bytes)."""
        X = self.X_train.toarray() if sp.issparse(self.X_train) else self.X_train
        m = torch.as_tensor(self.mask.toarray(), device=device) if self.mask is not None else None
        self._cache = (torch.tensor(np.array(X, dtype=np.float32), device=device), m)

    def block(self, cells, device):
        if self._cache is not None:
            idx = torch.as_tensor(cells, device=device)
            x, m = self._cache
            return x[idx], (m[idx] if m is not None else None)
        X = self.X_train[cells]
        X = X.toarray() if sp.issparse(X) else X
        x = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
        m = None
        if self.mask is not None:
            m = torch.as_tensor(self.mask[cells].toarray(), device=device)
        return x, m


def rna_counts(adata, layer="counts"):
    X = adata.layers[layer] if layer in adata.layers else adata.X
    X = sp.csr_matrix(X, dtype=np.float32)
    if X.nnz and not np.allclose(X.data, np.round(X.data)):
        raise ValueError("RNA layer does not look like raw integer counts")
    return X


def protein_counts(adata, key="protein_counts"):
    P = adata.obsm[key]
    names = np.asarray(P.columns) if hasattr(P, "columns") else np.array([f"p{i}" for i in range(P.shape[1])])
    P = np.asarray(P, dtype=np.float32)
    return P, names


def mask_rna(X: sp.csr_matrix, frac: float, seed: int):
    """Authors' protocol: hide `frac` of the nonzero entries (set to 0 in training) and sample the
    same number of true zeros as negatives for link prediction."""
    rng = np.random.default_rng(seed)
    X = X.tocoo()
    n_pos = int(round(frac * X.nnz))
    idx = rng.choice(X.nnz, n_pos, replace=False)
    held = {"rows": X.row[idx], "cols": X.col[idx], "values": X.data[idx].astype(np.float32)}
    keep = np.ones(X.nnz, bool)
    keep[idx] = False
    X_tr = sp.csr_matrix((X.data[keep], (X.row[keep], X.col[keep])), shape=X.shape, dtype=np.float32)
    # negatives: uniformly sampled cell-gene pairs that are zero in the full matrix
    Xc = X.tocsr()
    nr, nc = [], []
    while len(nr) < n_pos:
        r = rng.integers(0, X.shape[0], n_pos)
        c = rng.integers(0, X.shape[1], n_pos)
        z = np.asarray(Xc[r, c]).ravel() == 0
        nr.extend(r[z]); nc.extend(c[z])
    held["neg_rows"] = np.asarray(nr[:n_pos]); held["neg_cols"] = np.asarray(nc[:n_pos])
    return X_tr, held


def mask_dense(P: np.ndarray, frac: float, seed: int):
    """Proteins are rarely zero, so hide `frac` of ALL entries as missing (excluded from the loss)."""
    rng = np.random.default_rng(seed)
    n = int(round(frac * P.size))
    flat = rng.choice(P.size, n, replace=False)
    r, c = np.unravel_index(flat, P.shape)
    mask = sp.csr_matrix((np.ones(n, bool), (r, c)), shape=P.shape)
    return mask, {"rows": r, "cols": c, "values": P[r, c].astype(np.float32)}


def batch_ids(adata, keys, device, categories=None):
    """Integer codes per batch level. `categories` (from a reference model) keeps codes consistent."""
    ids, cats = [], []
    for i, k in enumerate(keys or []):
        col = adata.obs[k].astype(str)
        known = list(categories[i]) if categories else sorted(col.unique())
        extra = sorted(set(col) - set(known))
        known = known + extra
        codes = col.map({c: j for j, c in enumerate(known)}).to_numpy()
        ids.append(torch.as_tensor(codes, dtype=torch.long, device=device))
        cats.append(known)
    return ids, cats


def build(adata, use_protein, val_frac, seed, layer="counts", protein_key="protein_counts"):
    mods = {}
    X = rna_counts(adata, layer)
    held = None
    if val_frac > 0:
        X, held = mask_rna(X, val_frac, seed)
    mods["rna"] = ModalityData("rna", X, np.asarray(adata.var_names), heldout=held)
    if use_protein:
        P, names = protein_counts(adata, protein_key)
        mask, held_p = (None, None)
        if val_frac > 0:
            mask, held_p = mask_dense(P, val_frac, seed + 7)
        mods["protein"] = ModalityData("protein", P, names, heldout=held_p, mask=mask)
    return mods
