"""Multimodal SCENE: cells, genes and proteins as points in one latent space.

Follows the authors' SCENE (scene/model.py in SCENE-reproducibility) for the RNA part:
    eta_ij = psi_i + omega_j + sum_l g^(l)_{b_l(i), j} - ||z_i - w_j||      (Euclidean, not squared)
and adds
    * RNA likelihood variants: "zip" (authors' hurdle ZIP), "poisson", "nb", "zip_decoupled";
    * a protein (ADT) modality with a background + foreground negative binomial:
        mu_ij = exp(psi_i) * (exp(beta_j) + exp(omega_j - ||z_i - v_j||)),
      i.e. a protein far from a cell is at its ambient/background level beta_j.
Cells share z_i across modalities; every modality has its own feature positions, cell/feature
effects and batch effects (library sizes differ between RNA and ADT).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-10
RNA_LIKELIHOODS = ("zip", "poisson", "nb", "zip_decoupled")
PROTEIN_LIKELIHOODS = ("nb_bg", "nb")


@dataclass
class ModalitySpec:
    name: str                      # "rna" | "protein" | ...
    n_features: int
    likelihood: str                # see RNA_LIKELIHOODS / PROTEIN_LIKELIHOODS
    weight: float = 1.0            # multiplies this modality's log-likelihood
    batch_levels: list = field(default_factory=list)   # [(n_batches, "lowrank"|"full"), ...]
    rank: int = 2


class BatchEffects(nn.Module):
    """Additive per-feature batch offsets, one term per batch level (authors' parameterisation)."""

    def __init__(self, n_features, levels, rank):
        super().__init__()
        self.kinds = []
        self.U = nn.ParameterList()
        self.V = nn.ParameterList()
        for n_batches, kind in levels:
            if kind == "full":
                self.U.append(nn.Parameter(torch.zeros(n_batches, n_features)))
                self.V.append(nn.Parameter(torch.zeros(0)))
            elif kind == "lowrank":
                self.U.append(nn.Parameter(torch.randn(n_batches, rank) * 0.01))
                self.V.append(nn.Parameter(torch.randn(n_features, rank) * 0.01))
            else:
                raise ValueError(f"unknown batch parameterisation {kind}")
            self.kinds.append(kind)

    def forward(self, batch_ids):
        """batch_ids: list (per level) of LongTensor[n_cells_block] -> [n_block, n_features] or 0."""
        out = 0.0
        for kind, U, V, ids in zip(self.kinds, self.U, self.V, batch_ids or []):
            if ids is None:
                continue
            out = out + (U[ids] if kind == "full" else U[ids] @ V.T)
        return out

    def add_batch(self, level, init=None):
        """Append one new batch row at `level` (used when folding in a new donor). Returns its index."""
        U = self.U[level]
        row = torch.zeros(1, U.shape[1], device=U.device) if init is None else init
        self.U[level] = nn.Parameter(torch.cat([U.data, row], dim=0))
        return self.U[level].shape[0] - 1


class Modality(nn.Module):
    def __init__(self, spec: ModalitySpec, latent_dim: int, n_cells: int):
        super().__init__()
        self.spec = spec
        lik = spec.likelihood
        allowed = PROTEIN_LIKELIHOODS if spec.name == "protein" else RNA_LIKELIHOODS
        if lik not in allowed:
            raise ValueError(f"{spec.name}: likelihood must be one of {allowed}, got {lik}")
        F_ = spec.n_features
        self.W = nn.Parameter(torch.randn(F_, latent_dim))              # feature positions
        self.cell_re = nn.Parameter(torch.randn(n_cells, 1) * 0.01)      # psi_i
        self.feat_re = nn.Parameter(torch.randn(F_, 1) * 0.01)           # omega_j
        self.batch = BatchEffects(F_, spec.batch_levels, spec.rank)
        if lik in ("zip", "zip_decoupled"):
            self.raw_alpha = nn.Parameter(torch.log(torch.expm1(torch.tensor(1.0))))
        if lik == "zip_decoupled":
            # separate "is it detected?" effects; distance is shared
            self.cell_re_pi = nn.Parameter(torch.randn(n_cells, 1) * 0.01)
            self.feat_re_pi = nn.Parameter(torch.randn(F_, 1) * 0.01)
        if lik in ("nb", "nb_bg"):
            self.raw_theta = nn.Parameter(torch.full((F_,), math.log(math.expm1(10.0))))
        if lik == "nb_bg":
            self.bg = nn.Parameter(torch.full((F_, 1), -1.0))             # beta_j (log background)

    # --- parameters ------------------------------------------------------------------------
    @property
    def alpha(self):
        return F.softplus(self.raw_alpha)

    @property
    def theta(self):
        return F.softplus(self.raw_theta) + 1e-4

    def no_decay_params(self):
        """Effects, dispersions, alpha, background: no weight decay (as for the authors' random effects)."""
        names = ("cell_re", "feat_re", "cell_re_pi", "feat_re_pi", "raw_alpha", "raw_theta", "bg")
        return [getattr(self, n) for n in names if hasattr(self, n)]

    # --- distribution parameters for a block of cells ---------------------------------------
    def params(self, z_block, cells, batch_ids):
        """Return a dict of distribution parameters for cells (indices) x all features."""
        dist = torch.cdist(z_block, self.W)
        b = self.batch(batch_ids)
        eta = self.cell_re[cells] + self.feat_re.T + b - dist
        lik = self.spec.likelihood
        if lik == "zip":
            return {"lam": torch.exp(eta), "pi": torch.sigmoid(self.alpha * eta)}
        if lik == "zip_decoupled":
            eta_pi = self.cell_re_pi[cells] + self.feat_re_pi.T + b - dist
            return {"lam": torch.exp(eta), "pi": torch.sigmoid(self.alpha * eta_pi)}
        if lik == "poisson":
            return {"lam": torch.exp(eta)}
        if lik == "nb":
            return {"mu": torch.exp(eta), "theta": self.theta}
        if lik == "nb_bg":
            fg = torch.exp(eta)
            bgm = torch.exp(self.cell_re[cells] + self.bg.T + b)
            return {"mu": fg + bgm, "theta": self.theta, "fg": fg, "bg": bgm}
        raise AssertionError

    # --- log-likelihood per entry ------------------------------------------------------------
    def log_prob(self, p, x):
        lik = self.spec.likelihood
        pos = x > 0
        if lik in ("zip", "zip_decoupled"):
            lam, pi = p["lam"], p["pi"]
            log_expm1 = lam + torch.log(-torch.expm1(-lam) + EPS)
            lp_pos = torch.log(pi + EPS) + x * torch.log(lam + EPS) - torch.lgamma(x + 1) - log_expm1
            return torch.where(pos, lp_pos, torch.log(1 - pi + EPS))
        if lik == "poisson":
            lam = p["lam"]
            return x * torch.log(lam + EPS) - lam - torch.lgamma(x + 1)
        mu, theta = p["mu"], p["theta"]
        log_theta_mu = torch.log(theta + mu + EPS)
        return (theta * (torch.log(theta + EPS) - log_theta_mu) + x * (torch.log(mu + EPS) - log_theta_mu)
                + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1))

    def log_lik_sum(self, p, x, mask=None):
        """Sum of log p(x) over a block. Without a mask, the expensive terms (lgamma, log-expm1) are
        evaluated only at the nonzero counts, as in the authors' implementation (RNA is ~90% zeros).
        For x = 0: ZIP -> log(1-pi); Poisson -> -lam; NB -> theta*(log theta - log(theta+mu))."""
        if mask is not None:
            return self.log_prob(p, x).masked_fill(mask, 0.0).sum()
        lik = self.spec.likelihood
        pos = (x > 0).nonzero(as_tuple=True)
        xp = x[pos]
        if lik in ("zip", "zip_decoupled"):
            lam, pi = p["lam"], p["pi"]
            log1m_pi = torch.log(1 - pi + EPS)
            lp, pip = lam[pos], pi[pos]
            pos_term = (torch.log(pip + EPS) + xp * torch.log(lp + EPS) - torch.lgamma(xp + 1)
                        - (lp + torch.log(-torch.expm1(-lp) + EPS)) - log1m_pi[pos])
            return log1m_pi.sum() + pos_term.sum()
        if lik == "poisson":
            lam = p["lam"]
            return -lam.sum() + (xp * torch.log(lam[pos] + EPS) - torch.lgamma(xp + 1)).sum()
        mu, theta = p["mu"], p["theta"]
        log_tm = torch.log(theta + mu + EPS)
        zero_part = (theta * (torch.log(theta + EPS) - log_tm)).sum()
        th = theta.expand_as(mu)[pos]
        pos_term = (xp * (torch.log(mu[pos] + EPS) - log_tm[pos])
                    + torch.lgamma(xp + th) - torch.lgamma(th) - torch.lgamma(xp + 1))
        return zero_part + pos_term.sum()

    def prob_detect(self, p):
        """P(x > 0)."""
        lik = self.spec.likelihood
        if "pi" in p:
            return p["pi"]
        if lik == "poisson":
            return -torch.expm1(-p["lam"])
        mu, theta = p["mu"], p["theta"]
        return 1 - torch.exp(theta * (torch.log(theta + EPS) - torch.log(theta + mu + EPS)))

    def mean(self, p):
        """E[x]."""
        if "pi" in p:
            lam = p["lam"]
            return p["pi"] * lam / (-torch.expm1(-lam) + EPS)
        return p.get("lam", p.get("mu"))

    def mean_given_positive(self, p):
        """E[x | x > 0] (the authors' deviance uses this for held-out nonzero counts)."""
        if "pi" in p:
            lam = p["lam"]
            return lam / (-torch.expm1(-lam) + EPS)
        return self.mean(p) / (self.prob_detect(p) + EPS)


class MultiSCENE(nn.Module):
    def __init__(self, n_cells: int, latent_dim: int, specs: list[ModalitySpec]):
        super().__init__()
        self.latent_dim = latent_dim
        self.Z = nn.Parameter(torch.randn(n_cells, latent_dim))
        self.mods = nn.ModuleDict({s.name: Modality(s, latent_dim, n_cells) for s in specs})

    def param_groups(self, weight_decay=1e-3):
        no_decay, decay_pos, decay_batch = [], [self.Z], []
        for m in self.mods.values():
            no_decay += m.no_decay_params()
            decay_pos.append(m.W)
            decay_batch += [p for p in m.batch.parameters() if p.numel() > 0]
        groups = [{"params": no_decay, "weight_decay": 0.0},
                  {"params": decay_pos, "weight_decay": weight_decay}]
        if decay_batch:
            groups.append({"params": decay_batch, "weight_decay": weight_decay})
        return groups

    def block_log_lik(self, cells, x_blocks, batch_ids, masks=None):
        """Sum of log-likelihoods for a block of cells over all modalities.

        x_blocks: {modality: dense [n_block, n_features] float tensor}
        masks: {modality: bool tensor, True = entry held out (excluded)} or None
        """
        z = self.Z[cells]
        total, parts = 0.0, {}
        for name in x_blocks:          # only observed modalities (e.g. RNA-only fold-in of a multimodal model)
            mod = self.mods[name]
            p = mod.params(z, cells, batch_ids.get(name) if batch_ids else None)
            s = mod.log_lik_sum(p, x_blocks[name], masks.get(name) if masks else None)
            parts[name] = s.detach()
            total = total + mod.spec.weight * s
        return total, parts
