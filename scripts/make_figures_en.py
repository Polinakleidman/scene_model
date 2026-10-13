"""English versions of the Track A report figures for the slide deck. Run from scene_project/."""
import re
import sys
import warnings
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
T = Path("results_tables")
full = pd.read_csv(T / "full.csv"); bfull = pd.read_csv(T / "baseline_full.csv"); lodo = pd.read_csv(T / "lodo.csv")
pairs = pd.read_csv(T / "protein_pairs.csv"); psig = pd.read_csv(T / "protein_signal.csv")
LAB = {"pbmc": "celltype.l2", "neurips": "cell_type"}
LIKS = ["zip", "nb", "zip_decoupled"]; LIKN = {"zip": "ZIP", "nb": "NB", "zip_decoupled": "ZIP-decoupled"}
DSN = {"pbmc": "PBMC (217 proteins)", "neurips": "NeurIPS BMMC (134 proteins)"}
MODN = {"rna": "SCENE RNA", "rna+prot": "SCENE RNA + protein"}
BLUE, ORANGE = "#6c8ebf", "#d79b00"


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)


# H1: held-out NLL and detection AUC, 16D
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (ds, col, name) in zip(axes, [("pbmc", "rna_heldout_nll", "PBMC: held-out NLL (lower = better)"),
                                       ("neurips", "rna_heldout_nll", "NeurIPS: held-out NLL (lower = better)"),
                                       ("pbmc", "rna_auc_roc", "PBMC: detection AUC")]):
    d = full[(full.dataset == ds) & (full.dim == 16)]
    lows, highs = [], []
    for k, mod in enumerate(["rna", "rna+prot"]):
        g = d[d.modality == mod].groupby("likelihood")[col]
        m_, s_ = np.array([g.mean()[l] for l in LIKS]), np.nan_to_num(np.array([g.std()[l] for l in LIKS]))
        ax.bar(np.arange(3) + (k - 0.5) * 0.35, m_, 0.35, yerr=s_, label=MODN[mod], color=[BLUE, ORANGE][k])
        lows.append((m_ - s_).min()); highs.append((m_ + s_).max())
    lo, hi = min(lows), max(highs); pad = 0.25 * (hi - lo) + 1e-4
    ax.set_ylim(lo - pad, hi + pad); ax.set_xticks(range(3)); ax.set_xticklabels([LIKN[l] for l in LIKS]); ax.set_title(name, fontsize=12)
axes[0].legend(frameon=False, fontsize=10)
fig.tight_layout(); save(fig, "h1_likelihood.png")

# H2: NMI on fine cell types
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for ax, (ds, lab) in zip(axes, LAB.items()):
    d = full[(full.dataset == ds) & (full.dim == 16)]
    for k, mod in enumerate(["rna", "rna+prot"]):
        g = d[d.modality == mod].groupby("likelihood")[f"{lab}_nmi"]
        ax.bar(np.arange(3) + (k - 0.5) * 0.35, [g.mean()[l] for l in LIKS], 0.35, yerr=[g.std()[l] for l in LIKS],
               color=[BLUE, ORANGE][k], label=MODN[mod])
    for m, col in [("scvi", "#82b366"), ("totalvi", "#9673a6")]:
        v = bfull[(bfull.dataset == ds) & (bfull.model == m) & (bfull.label == lab)].nmi.mean()
        ax.axhline(v, color=col, ls="--", lw=1.5, label={"scvi": "scVI", "totalvi": "totalVI"}[m])
    ax.set_xticks(range(3)); ax.set_xticklabels([LIKN[l] for l in LIKS]); ax.set_ylim(0.55, 0.73)
    ax.set_title(f"{ds.upper()}: NMI on fine cell types", fontsize=12)
axes[0].legend(frameon=False, fontsize=10, loc="lower left")
fig.tight_layout(); save(fig, "h2_nmi.png")

# H3: own-gene percentile curves
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for ax, ds in zip(axes, ["pbmc", "neurips"]):
    for dim, ls in [(16, "-"), (3, ":")]:
        for lik, col in [("zip", BLUE), ("nb", "#b85450")]:
            r = np.sort(pairs[(pairs.dataset == ds) & (pairs.dim == dim) & (pairs.likelihood == lik)].rank_pct)
            ax.plot(r, np.arange(1, len(r) + 1) / len(r), ls, color=col, lw=2, label=f"{LIKN[lik]} {dim}D")
    xs = np.r_[np.linspace(0, 0.01, 200), np.logspace(-2, 0, 400)]; ax.plot(xs, xs, color="grey", lw=1, label="random")
    ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(0, 1)
    ax.set_xlabel("percentile of own gene (0 = nearest)"); ax.set_ylabel("fraction of proteins"); ax.set_title(DSN[ds], fontsize=12)
axes[0].legend(frameon=False, fontsize=10)
fig.tight_layout(); save(fig, "h3_rank.png")

# H3: signal share vs own-gene percentile
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for ax, ds in zip(axes, ["pbmc", "neurips"]):
    a = psig[psig.dataset == ds].groupby(["protein", "gene"]).mean(numeric_only=True).reset_index()
    rho, p = spearmanr(a.fg_share, a.rank_pct)
    ax.scatter(a.fg_share, a.rank_pct, s=16, alpha=0.7, color=BLUE)
    for _, r in a[(a.rank_pct < 0.0005) | (a.rank_pct > 0.45)].iterrows():
        ax.annotate(r.protein, (r.fg_share, r.rank_pct), fontsize=8)
    ax.set_yscale("symlog", linthresh=0.001); ax.set_xlabel("signal share of protein count (1 − background)")
    ax.set_ylabel("percentile of own gene"); ax.set_title(f"{ds.upper()}: Spearman ρ = {rho:+.2f}", fontsize=12)
fig.tight_layout(); save(fig, "h3_signal.png")

# H4/H5: leave-one-donor-out label transfer
d = lodo[lodo.label == "celltype.l2"].copy(); d["key"] = d.model + " / " + d.embedding
keys = ["SCENE rna zip / aggregate", "SCENE rna zip / foldin", "SCENE rna+prot nb / foldin", "scvi / scArches", "totalvi / scArches"]
names = ["SCENE RNA\naggregation", "SCENE RNA\nfold-in", "SCENE RNA+prot\nfold-in", "scVI\nscArches", "totalVI\nscArches"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
for ax, metric, title in [(axes[0], "accuracy", "accuracy"), (axes[1], "macro_f1", "macro-F1")]:
    for i, k in enumerate(keys):
        v = d[d.key == k].sort_values("donor")[metric].to_numpy()
        ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=20, alpha=0.7, color=ORANGE if "SCENE" in k else "#7a7a7a")
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color="k", lw=2)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(names, fontsize=10)
    ax.set_title(f"New donor, 30 fine types: {title}", fontsize=12)
fig.tight_layout(); save(fig, "h4_lodo.png")

# Joint-space UMAP
import umap
emb = ad.read_h5ad("results/pbmc_full_rna+prot_nb_d16_s0/embedding.h5ad")
Z = emb.obsm["X_scene"]; W = emb.varm["W_rna"]; P = emb.uns["protein_positions"]
V = P[[f"w{i}" for i in range(Z.shape[1])]].to_numpy()
show_pairs = [("CD3", "CD3E"), ("CD4", "CD4"), ("CD8", "CD8A"), ("CD19", "CD19"), ("CD20", "MS4A1"), ("CD14", "CD14"),
              ("CD16", "FCGR3A"), ("CD56", "NCAM1"), ("CD127", "IL7R"), ("HLA-DR", "HLA-DRA"), ("CD123", "IL3RA"), ("CD1c", "CD1C")]
genes = list(emb.var_names); pnames = [re.sub(r"-\d+$", "", str(n)) for n in P.index]
show_pairs = [(p_, g) for p_, g in show_pairs if p_ in pnames and g in genes]
rng = np.random.default_rng(0)
ci = rng.choice(len(Z), 6000, replace=False)
gi = np.unique(np.r_[rng.choice(len(W), 3000, replace=False), [genes.index(g) for _, g in show_pairs]])
pi_ = [pnames.index(p_) for p_, _ in show_pairs]
E = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(np.vstack([Z[ci], W[gi], V[pi_]]))
ec, eg, ep = E[:len(ci)], E[len(ci):len(ci) + len(gi)], E[len(ci) + len(gi):]
lab = emb.obs["celltype.l1"].astype(str).to_numpy()[ci]
fig, ax = plt.subplots(figsize=(10, 7.5))
ax.scatter(eg[:, 0], eg[:, 1], s=1, color="lightgrey", alpha=0.5, label="genes (subsample)")
for i, l in enumerate(sorted(set(lab))):
    m = lab == l; ax.scatter(ec[m, 0], ec[m, 1], s=2, alpha=0.4, label=l, color=plt.cm.tab10(i))
for k, (p_, g) in enumerate(show_pairs):
    gv = eg[list(gi).index(genes.index(g))]; pv = ep[k]
    ax.plot([pv[0], gv[0]], [pv[1], gv[1]], color="k", lw=0.8)
    ax.scatter(*pv, color="k", s=28, zorder=5); ax.scatter(*gv, color="red", s=20, zorder=5)
    ax.annotate(p_, pv, fontsize=9, fontweight="bold"); ax.annotate(g, gv, fontsize=8, color="red", xytext=(3, -9), textcoords="offset points")
ax.legend(frameon=False, fontsize=9, markerscale=4, loc="upper right"); ax.set_xticks([]); ax.set_yticks([])
ax.spines[["left", "bottom"]].set_visible(False)
ax.set_title("Cells (by type), genes (grey), proteins (black) and their genes (red)", fontsize=12)
save(fig, "h3_umap.png")
print("done")
