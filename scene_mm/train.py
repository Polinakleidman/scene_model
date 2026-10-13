"""Training and held-out evaluation for multimodal SCENE."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from . import data as D
from .model import ModalitySpec, MultiSCENE


@dataclass
class Config:
    latent_dim: int = 16
    epochs: int = 600
    lr: float = 0.05
    weight_decay: float = 1e-3
    rna_likelihood: str = "zip"
    use_protein: bool = False
    protein_likelihood: str = "nb_bg"
    protein_weight: float = 1.0
    batch_keys: list = field(default_factory=list)
    batch_kind: str = "lowrank"          # "lowrank" | "full"
    rank: int = 2
    block_size: int = 2048               # cells per forward block (memory only)
    minibatch: bool = False              # False: accumulate all blocks -> one full-batch step (authors' default)
    val_frac: float = 0.1
    seed: int = 0
    device: str = "cpu"
    eval_every: int = 50
    layer: str = "counts"
    cache_gb: float = 12.0              # keep dense counts on the device if they fit in this budget


def _specs(mods, cfg, n_batches):
    levels = [(n, cfg.batch_kind) for n in n_batches]
    specs = [ModalitySpec("rna", mods["rna"].n_features, cfg.rna_likelihood, 1.0, levels, cfg.rank)]
    if "protein" in mods:
        specs.append(ModalitySpec("protein", mods["protein"].n_features, cfg.protein_likelihood,
                                  cfg.protein_weight, levels, cfg.rank))
    return specs


def _blocks(n, size, rng=None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    return [order[s:s + size] for s in range(0, n, size)]


def _block_batch_ids(ids, cells_t):
    return [i[cells_t] for i in ids] if ids else None


def fit(adata, cfg: Config, log=print):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = cfg.device
    mods = D.build(adata, cfg.use_protein, cfg.val_frac, cfg.seed, layer=cfg.layer)
    ids, cats = D.batch_ids(adata, cfg.batch_keys, dev)
    model = MultiSCENE(adata.n_obs, cfg.latent_dim, _specs(mods, cfg, [len(c) for c in cats])).to(dev)
    opt = torch.optim.AdamW(model.param_groups(cfg.weight_decay), lr=cfg.lr)
    n = adata.n_obs
    n_entries = sum(adata.n_obs * m.n_features for m in mods.values())
    if n_entries * 4 / 1e9 <= cfg.cache_gb:
        for m in mods.values():
            m.cache_on(dev)
    history = {"epoch": [], "loss": [], "time_s": []}
    t0 = time.time()
    for epoch in range(cfg.epochs):
        # full-batch mode: block order does not change the gradient, so blocks stay fixed
        rng = np.random.default_rng(cfg.seed * 100003 + epoch) if cfg.minibatch else None
        blocks = _blocks(n, cfg.block_size, rng)
        if not cfg.minibatch:
            opt.zero_grad(set_to_none=True)
        total = 0.0
        for cells in blocks:
            cells_t = torch.as_tensor(cells, device=dev)
            xs, ms = {}, {}
            for name, m in mods.items():
                xs[name], ms[name] = m.block(cells, dev)
            ll, _ = model.block_log_lik(cells_t, xs, {k: _block_batch_ids(ids, cells_t) for k in mods}, ms)
            if cfg.minibatch:
                opt.zero_grad(set_to_none=True)
                (-ll * (n / len(cells)) / n_entries).backward()
                opt.step()
            else:
                (-ll / n_entries).backward()
            total += float(ll.detach())
        if not cfg.minibatch:
            opt.step()
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            history["epoch"].append(epoch)
            history["loss"].append(-total / n_entries)
            history["time_s"].append(round(time.time() - t0, 1))
            log(f"epoch {epoch:4d}  loss {-total / n_entries:.4f}  ({time.time() - t0:.0f}s)")
    metrics = evaluate_heldout(model, mods, ids, cfg) if cfg.val_frac > 0 else {}
    metrics["train_time_s"] = round(time.time() - t0, 1)
    return model, mods, (ids, cats), history, metrics


# ------------------------------------------------------------------------------------------
@torch.no_grad()
def _params_at(model, name, rows, cols, ids, cfg):
    """Distribution parameters at specific (cell, feature) pairs, computed block-wise over cells."""
    mod = model.mods[name]
    out = {}
    order = np.argsort(rows, kind="stable")
    rows_s, cols_s = rows[order], cols[order]
    uniq, start = np.unique(rows_s, return_index=True)
    chunks = []
    for b in range(0, len(uniq), cfg.block_size):
        cells = uniq[b:b + cfg.block_size]
        lo = start[b]
        hi = start[b + cfg.block_size] if b + cfg.block_size < len(uniq) else len(rows_s)
        cells_t = torch.as_tensor(cells, device=cfg.device)
        p = mod.params(model.Z[cells_t], cells_t, _block_batch_ids(ids, cells_t))
        local = np.searchsorted(cells, rows_s[lo:hi])
        r = torch.as_tensor(local, device=cfg.device)
        c = torch.as_tensor(cols_s[lo:hi], device=cfg.device)
        chunks.append({k: (v[r, c] if v.dim() == 2 else v[c]) for k, v in p.items()})
    for k in chunks[0]:
        vals = torch.cat([ch[k] for ch in chunks]).cpu().numpy()
        out[k] = np.empty_like(vals)
        out[k][order] = vals
    return {k: torch.as_tensor(v) for k, v in out.items()}


def _max_f1(y, s):
    p, r, _ = precision_recall_curve(y, s)
    return float(np.max(2 * p * r / (p + r + 1e-12)))


@torch.no_grad()
def evaluate_heldout(model, mods, ids, cfg):
    res = {}
    rna = mods["rna"].heldout
    if rna is not None:
        mod = model.mods["rna"]
        pp = _params_at(model, "rna", rna["rows"], rna["cols"], ids, cfg)
        pn = _params_at(model, "rna", rna["neg_rows"], rna["neg_cols"], ids, cfg)
        x = torch.as_tensor(rna["values"])
        s_pos, s_neg = mod.prob_detect(pp).numpy(), mod.prob_detect(pn).numpy()
        y = np.r_[np.ones_like(s_pos), np.zeros_like(s_neg)]
        s = np.r_[s_pos, s_neg]
        mu = mod.mean_given_positive(pp).clamp_min(1e-8)
        dev = 2 * (x * torch.log(x / mu) - (x - mu))
        res["rna"] = {
            "auc_roc": float(roc_auc_score(y, s)),
            "pr_auc": float(average_precision_score(y, s)),
            "max_f1": _max_f1(y, s),
            "poisson_deviance": float(dev.mean()),
            "heldout_nll_pos": float(-mod.log_prob(pp, x).mean()),
            "heldout_nll_neg": float(-mod.log_prob(pn, torch.zeros(len(s_neg))).mean()),
        }
        r = res["rna"]
        r["heldout_nll"] = 0.5 * (r["heldout_nll_pos"] + r["heldout_nll_neg"])
    if "protein" in mods and mods["protein"].heldout is not None:
        h = mods["protein"].heldout
        mod = model.mods["protein"]
        pp = _params_at(model, "protein", h["rows"], h["cols"], ids, cfg)
        x = torch.as_tensor(h["values"])
        mu = mod.mean(pp).clamp_min(1e-8)
        res["protein"] = {
            "heldout_nll": float(-mod.log_prob(pp, x).mean()),
            "pearson_log1p": float(np.corrcoef(np.log1p(x.numpy()), np.log1p(mu.numpy()))[0, 1]),
            "poisson_deviance": float((2 * (torch.where(x > 0, x * torch.log(x / mu), torch.zeros_like(x)) - (x - mu))).mean()),
        }
    return res


# ------------------------------------------------------------------------------------------
def load_run(run_dir, adata, device="cpu"):
    """Rebuild a trained model from results/<run_id>/ (train_log.json + model.pt).

    The data are rebuilt with the same seed, so the held-out entries are exactly those of training.
    Returns (model, mods, batch_ids, cfg)."""
    run_dir = Path(run_dir)
    c = json.loads((run_dir / "train_log.json").read_text())["config"]
    cfg = Config(**{k: v for k, v in c.items() if k in Config.__dataclass_fields__})
    cfg.device = device
    mods = D.build(adata, cfg.use_protein, cfg.val_frac, cfg.seed, layer=cfg.layer)
    ids, cats = D.batch_ids(adata, cfg.batch_keys, device)
    model = MultiSCENE(adata.n_obs, cfg.latent_dim, _specs(mods, cfg, [len(x) for x in cats])).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    return model, mods, ids, cfg


def export(model, mods, cats, adata, cfg, history, metrics, out_dir):
    """Write embedding.h5ad (cells in obsm, genes in varm, proteins in uns) and metrics.json."""
    import anndata as ad
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = ad.AnnData(obs=adata.obs.copy(), var=adata.var[[]].copy())
    emb.obsm["X_scene"] = model.Z.detach().cpu().numpy()
    rna = model.mods["rna"]
    emb.varm["W_rna"] = rna.W.detach().cpu().numpy()
    emb.var["omega_rna"] = rna.feat_re.detach().cpu().numpy().ravel()
    emb.obs["psi_rna"] = rna.cell_re.detach().cpu().numpy().ravel()
    if hasattr(rna, "raw_theta"):
        emb.var["theta_rna"] = rna.theta.detach().cpu().numpy()
    if "protein" in model.mods:
        pm = model.mods["protein"]
        cols = {f"w{i}": pm.W.detach().cpu().numpy()[:, i] for i in range(pm.W.shape[1])}
        dfp = pd.DataFrame(cols, index=mods["protein"].feature_names)
        dfp["omega"] = pm.feat_re.detach().cpu().numpy().ravel()
        dfp["theta"] = pm.theta.detach().cpu().numpy()
        if hasattr(pm, "bg"):
            dfp["log_background"] = pm.bg.detach().cpu().numpy().ravel()
        emb.uns["protein_positions"] = dfp
        emb.obs["psi_protein"] = pm.cell_re.detach().cpu().numpy().ravel()
    emb.uns["config"] = {k: (v if not isinstance(v, list) else ",".join(map(str, v))) for k, v in asdict(cfg).items()}
    emb.uns["batch_categories"] = {k: list(c) for k, c in zip(cfg.batch_keys, cats)} if cfg.batch_keys else {}
    emb.write_h5ad(out / "embedding.h5ad")
    torch.save(model.state_dict(), out / "model.pt")
    (out / "train_log.json").write_text(json.dumps({"config": asdict(cfg), "heldout": metrics, "history": history}, indent=2))
    return out
