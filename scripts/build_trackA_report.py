"""Build the ML-track report: trackA_report.ipynb -> trackA_report.html (single file, code hidden).

Needs: results_tables/*.csv (scripts/collect_results.py, analyze_likelihood.py, analyze_proteins.py)
and results/pbmc_full_rna+prot_nb_d16_s0/embedding.h5ad for the joint-space picture.
Run from scene_project/:  python scripts/build_trackA_report.py
"""
import subprocess
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJ = Path(__file__).resolve().parents[1]
NB = PROJ / "trackA_report.ipynb"
HTML = PROJ / "trackA_report.html"
cells = []


def md(t):
    cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))


def code(t):
    cells.append(nbf.v4.new_code_cell(t.strip("\n")))


md(r"""
# Track A: multimodal SCENE — results

**Project:** *Joint cell–gene–protein embeddings for interpreting drug responses: a stress test of SCENE* (course CBIO7215).
**Track A (ML):**
1. Add surface proteins (CITE-seq) to SCENE as a third type of point.
2. Test other likelihoods for RNA.
3. Embed new donors without retraining.
4. Compare with totalVI and scVI.

Code: `scene_mm/`. The model and the **hypotheses written down before any results** are in `scene_mm/README.md`.

### Summary
- **The model works and matches the authors' code:** RNA-only ZIP gives the same AUC, PR-AUC and F1 as the authors' implementation to the 3rd–4th decimal.
- **Proteins land next to their own genes:** the median percentile of the own gene is 0.4–3% among ~14–21k genes (random placement: 50%). Lineage markers work best (CD16 ↔ *FCGR3A*, CD20 ↔ *MS4A1*, CD127 ↔ *IL7R*); proteins with no signal in blood work worst. The signal share of a protein's count explains this on both datasets (Spearman ≈ −0.65).
- **Proteins improve clustering of fine cell types** moderately, and **clearly improve label transfer to new donors:** fold-in 0.83 → 0.90.
- **NB is not worse than ZIP:** its "bad" deviance is an artefact of a metric that assumes variance = mean. On held-out likelihood, the right metric across distribution families, NB is better.
- **Against totalVI:** comparable accuracy, not better (0.90 vs 0.91; a tie with protein weight 3, section 7). SCENE's advantage is interpretability: genes and proteins share the space with the cells.
""")

code(r"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, anndata as ad
from scipy.stats import spearmanr
from IPython.display import display, Markdown
T = Path("results_tables")
full = pd.read_csv(T / "full.csv"); bfull = pd.read_csv(T / "baseline_full.csv"); lodo = pd.read_csv(T / "lodo.csv")
pairs = pd.read_csv(T / "protein_pairs.csv"); calib = pd.read_csv(T / "likelihood_calibration.csv")
theta = pd.read_csv(T / "likelihood_theta.csv"); psig = pd.read_csv(T / "protein_signal.csv")
LAB = {"pbmc": "celltype.l2", "neurips": "cell_type"}
LIKS = ["zip", "nb", "zip_decoupled"]; LIKN = {"zip": "ZIP", "nb": "NB", "zip_decoupled": "ZIP-decoupled"}
DSN = {"pbmc": "PBMC (30k cells, 217 proteins)", "neurips": "NeurIPS BMMC (30k cells, 134 proteins)"}
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
def ms(s): return f"{s.mean():.3f} ± {s.std():.3f}"
""")

md(r"""
## 1. What was built

**Model.** Each cell has one point $z_i$, shared by all modalities. Genes have their own points $w_j$, proteins $v_k$. The cell–gene affinity is
$$\eta_{ij} = \psi_i + \rho_j + \textstyle\sum_\ell g^{(\ell)}_{b_\ell(i),j} - \lVert z_i - w_j\rVert$$
Here $\psi_i$ and $\rho_j$ are scalars: cell and gene effects ($\omega_j$ in the paper). The distance is Euclidean, not squared, as in the authors' code.

| Modality | Likelihood |
|---|---|
| RNA `zip` | as in the paper: $P(x>0)=\sigma(\alpha\eta)$, positive counts zero-truncated Poisson with $\lambda = e^\eta$ |
| RNA `nb` | negative binomial with mean $e^\eta$ and a dispersion $\theta_j$ per gene |
| RNA `zip_decoupled` | like `zip`, but detection has its own cell and gene effects |
| **Proteins** `nb_bg` | NB with mean $\mu_{ik} = e^{\psi_i}\big(e^{\beta_k} + e^{\rho_k - \lVert z_i - v_k\rVert}\big)$: background + signal |

The background $\beta_k$ models ambient antibody that sticks to every cell. A protein far from a cell stays at background level, not at zero.

**New donors:**
- **aggregation** (no training): a cell is the weighted mean of the points of its genes and proteins, weight = count / mean count of the feature;
- **fold-in**: everything feature-related is frozen; only the new cells and the new donor's batch row are trained (200 epochs).

**Data:**
- PBMC CITE-seq (Hao 2021): the same 30k-cell subset as in the reproduction, 8 donors, 30 fine types `celltype.l2`;
- NeurIPS 2021 BMMC CITE-seq: 30k random cells, 45 types, 12 batches.

**Grid:** 132 runs on two RTX 5000 Ada GPUs, ~7 h.
- Full runs: {RNA, RNA+protein} × 3 likelihoods × {3D, 16D} × 2 datasets × 3 seeds.
- Leave-one-donor-out (LODO) on PBMC: 8 donors × 4 models.
- scVI and totalVI.

**Check against the authors' code.** RNA-only ZIP, 2000 PBMC cells, 16D, 600 epochs:

| | AUC | PR-AUC | F1 | deviance |
|---|---|---|---|---|
| authors' code, seeds 0/1 | 0.9236 / 0.9236 | 0.9194 / 0.9192 | 0.8477 / 0.8481 | 0.538 / 0.575 |
| `scene_mm`, seeds 0/1 | 0.9235 / 0.9235 | 0.9193 / 0.9192 | 0.8477 / 0.8479 | 0.583 / 0.546 |
""")

# ---------------------------------------------------------------- H1
md(r"""
## 2. H1 — Likelihood for RNA

*Hypothesis (before results):* NB gives a better held-out log-likelihood, AUC is similar for all, and the decoupled variant improves AUC slightly.

10% of the non-zero counts are hidden from training, plus the same number of true zeros. Metrics:
- **NLL**: negative log-likelihood of the hidden values. Lower is better. **The only metric that compares distribution families fairly.**
- **AUC**: does the model separate hidden non-zeros from zeros using $P(x > 0)$.
- **Poisson deviance**: the authors' metric for count size, using $E[x \mid x > 0]$.

Mean ± standard deviation over 3 seeds.
""")
code(r"""
t = full.assign(lik=full.likelihood.map(LIKN)).groupby(["dataset", "dim", "modality", "lik"]).agg(
    NLL=("rna_heldout_nll", ms), AUC=("rna_auc_roc", ms), deviance=("rna_poisson_deviance", ms))
t
""")
code(r"""
fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
for r, ds in enumerate(["pbmc", "neurips"]):
    d = full[(full.dataset == ds) & (full.dim == 16)]
    for c, (col, name) in enumerate([("rna_heldout_nll", "held-out NLL (lower = better)"), ("rna_auc_roc", "detection AUC"),
                                     ("rna_poisson_deviance", "Poisson deviance (lower = better)")]):
        ax = axes[r, c]
        lows, highs = [], []
        for k, mod in enumerate(["rna", "rna+prot"]):
            g = d[d.modality == mod].groupby("likelihood")[col]
            m_, s_ = np.array([g.mean()[l] for l in LIKS]), np.nan_to_num(np.array([g.std()[l] for l in LIKS]))
            ax.bar(np.arange(3) + (k - 0.5) * 0.35, m_, 0.35, yerr=s_,
                   label={"rna": "RNA", "rna+prot": "RNA + protein"}[mod], color=["#6c8ebf", "#d79b00"][k])
            lows.append((m_ - s_).min()); highs.append((m_ + s_).max())
        lo, hi = min(lows), max(highs); pad = 0.25 * (hi - lo) + 1e-4
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(range(3)); ax.set_xticklabels([LIKN[l] for l in LIKS]); ax.set_title(f"{ds.upper()}, 16D — {name}", fontsize=8)
axes[0, 0].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.show()
""")
md(r"""
**Metrics in short:**
- **NB is better on NLL** on both datasets: PBMC 1.173 vs 1.190 for ZIP; NeurIPS 1.226 vs 1.307.
- **AUC is the same** for all three: ~0.926 and ~0.897.
- **NB's deviance is clearly worse.**
- Decoupled did not improve AUC. That part of the hypothesis is **not supported**.

### Why NB has worse deviance but better NLL

The trained models (PBMC, RNA only, 16D, seed 0) were re-evaluated on exactly the same hidden values, with the metrics split by the size of the observed count:
""")
code(r"""
c = calib[calib.bin != "zeros (true negatives)"].copy()
show = c.pivot(index="bin", columns="likelihood", values=["n", "mean_obs", "mean_pred_given_pos", "mean_deviance", "share_of_total_deviance", "mean_nll"])
show = show.reindex(["1", "2", "3-4", "5-9", "10-19", "20+"])
show.columns = [f"{a} [{b}]" for a, b in show.columns]
display(show[["n [zip]", "mean_obs [zip]", "mean_pred_given_pos [zip]", "mean_pred_given_pos [nb]",
              "mean_deviance [zip]", "mean_deviance [nb]", "share_of_total_deviance [nb]", "mean_nll [zip]", "mean_nll [nb]"]].round(3))
fig, axes = plt.subplots(1, 3, figsize=(11, 3))
order = ["1", "2", "3-4", "5-9", "10-19", "20+"]
for lik, col in [("zip", "#6c8ebf"), ("nb", "#b85450")]:
    d = c[c.likelihood == lik].set_index("bin").reindex(order)
    axes[0].plot(order, d.mean_pred_given_pos / d.mean_obs, "o-", color=col, label=LIKN[lik])
    axes[1].plot(order, d.mean_deviance, "o-", color=col, label=LIKN[lik])
    axes[2].plot(order, d.mean_nll, "o-", color=col, label=LIKN[lik])
axes[0].axhline(1, color="grey", lw=0.8); axes[0].set_title("predicted / observed (1 = perfect)", fontsize=8)
axes[1].set_title("Poisson deviance by bin", fontsize=8); axes[2].set_title("NLL by bin", fontsize=8)
for a in axes: a.set_xlabel("observed count")
axes[0].legend(frameon=False); plt.tight_layout(); plt.show()
display(Markdown(f"NB dispersion per gene: median $\\theta$ = {theta.theta.median():.1f}; {100 * (theta.theta < 1).mean():.0f}% of genes have $\\theta < 1$ (strong over-dispersion)."))
""")
md(r"""
**Conclusion for H1**
- **On counts of 1** (62% of all hidden non-zeros) NB beats ZIP on both metrics: NLL 1.87 vs 1.93. On true zeros the models are equal (NLL 0.069).
- **On counts of 2–9** NB is slightly worse on NLL (e.g. 2.56 vs 2.50 for twos), and from 5 upwards also on deviance.
- NB's main loss on deviance is **large counts (≥ 20, 3.4% of non-zeros), which make up ~48% of its total deviance.** There NB under-predicts the mean (~39 vs an observed 51; ZIP 48). It explains a large count with its **heavy tail** rather than by raising the mean.
- Poisson deviance assumes variance = mean, so it heavily penalises an under-predicted mean.
- Log-likelihood accounts for the tail, and on it **NB is better on the same large counts** (4.32 vs 4.66).
- Over all hidden values, NB wins on NLL thanks to the most frequent counts (ones) and the large counts.

**Implication:** the authors' Poisson deviance is fine for comparisons within the Poisson family, but not between ZIP and NB. **H1 is supported** for NB and **not supported** for the decoupled variant.
""")

# ---------------------------------------------------------------- H2
md(r"""
## 3. H2 — Do proteins help separate cell types?

*Hypothesis:* proteins improve clustering of fine cell types (NMI and ARI).

KMeans with $k$ = number of types on the cell embedding, compared with the labels: PBMC `celltype.l2` (30 types), NeurIPS `cell_type` (45 types). Mean ± standard deviation over 3 seeds, 16D. For comparison: scVI and totalVI (16D, 3 seeds).
""")
code(r"""
rows = []
for ds, lab in LAB.items():
    d = full[(full.dataset == ds) & (full.dim == 16)]
    for mod in ["rna", "rna+prot"]:
        for lik in LIKS:
            s = d[(d.modality == mod) & (d.likelihood == lik)]
            rows.append({"dataset": ds, "model": f"SCENE {'RNA' if mod == 'rna' else 'RNA+protein'} {LIKN[lik]}",
                         "NMI": ms(s[f"{lab}_nmi"]), "ARI": ms(s[f"{lab}_ari"])})
    for m in ["scvi", "totalvi"]:
        s = bfull[(bfull.dataset == ds) & (bfull.model == m) & (bfull.label == lab)]
        rows.append({"dataset": ds, "model": {"scvi": "scVI (RNA)", "totalvi": "totalVI (RNA+protein)"}[m], "NMI": ms(s.nmi), "ARI": ms(s.ari)})
pd.DataFrame(rows).set_index(["dataset", "model"])
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for ax, (ds, lab) in zip(axes, LAB.items()):
    d = full[(full.dataset == ds) & (full.dim == 16)]
    for k, mod in enumerate(["rna", "rna+prot"]):
        g = d[d.modality == mod].groupby("likelihood")[f"{lab}_nmi"]
        ax.bar(np.arange(3) + (k - 0.5) * 0.35, [g.mean()[l] for l in LIKS], 0.35, yerr=[g.std()[l] for l in LIKS],
               color=["#6c8ebf", "#d79b00"][k], label={"rna": "SCENE RNA", "rna+prot": "SCENE RNA + protein"}[mod])
    for j, (m, col) in enumerate([("scvi", "#82b366"), ("totalvi", "#9673a6")]):
        v = bfull[(bfull.dataset == ds) & (bfull.model == m) & (bfull.label == lab)].nmi.mean()
        ax.axhline(v, color=col, ls="--", lw=1.2, label={"scvi": "scVI", "totalvi": "totalVI"}[m])
    ax.set_xticks(range(3)); ax.set_xticklabels([LIKN[l] for l in LIKS]); ax.set_ylim(0.55, 0.73)
    ax.set_title(f"{ds.upper()}: NMI on fine types ({lab})", fontsize=8)
axes[0].legend(frameon=False, fontsize=7, loc="lower left"); plt.tight_layout(); plt.show()
""")
md(r"""
**Conclusion for H2: supported, moderate effect.**
- Proteins raise NMI for all three likelihoods on both datasets. PBMC: ZIP 0.645 → 0.667, NB 0.617 → 0.679. NeurIPS: +0.01–0.02.
- The best multimodal SCENE is level with scVI and slightly below totalVI (with protein weight 3 it reaches 0.702 vs 0.707, section 7).
- Proteins help most on PBMC: it has 217 proteins, and T and NK subtypes are defined by surface markers.
""")

# ---------------------------------------------------------------- H3
md(r"""
## 4. H3 — Each protein next to its own gene

*Hypothesis:* in the joint space a protein is closer to its own gene than a random gene is.

For each protein with a known gene (hand-made ADT → gene dictionary, ~110 pairs per dataset), all genes are ranked by distance to the protein. The metric is the **percentile of the own gene** in that list: 0 = the own gene is the nearest, 0.5 = like a random gene.
""")
code(r"""
s = full[full.modality == "rna+prot"].assign(lik=lambda d: d.likelihood.map(LIKN)).groupby(["dataset", "dim", "lik"]).agg(
    pairs=("pg_n_pairs", "first"), median_percentile=("pg_median_rank", ms), share_top1=("pg_top1", ms), share_top5=("pg_top5", ms))
s
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
for ax, ds in zip(axes, ["pbmc", "neurips"]):
    for dim, ls in [(16, "-"), (3, ":")]:
        for lik, col in [("zip", "#6c8ebf"), ("nb", "#b85450")]:
            r = np.sort(pairs[(pairs.dataset == ds) & (pairs.dim == dim) & (pairs.likelihood == lik)].rank_pct)
            ax.plot(r, np.arange(1, len(r) + 1) / len(r), ls, color=col, label=f"{LIKN[lik]} {dim}D")
    xs = np.r_[np.linspace(0, 0.01, 200), np.logspace(-2, 0, 400)]; ax.plot(xs, xs, color="grey", lw=0.8, label="random")
    ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(0, 1)
    ax.set_xlabel("percentile of own gene (0 = nearest)"); ax.set_ylabel("fraction of proteins"); ax.set_title(DSN[ds], fontsize=8)
axes[0].legend(frameon=False, fontsize=7); plt.tight_layout(); plt.show()
""")
md(r"""
Cumulative curves over all pairs (all seeds). A curve hugging the top-left corner means almost every protein sits next to its gene. The grey curve is random placement (the x-axis is linear below 1% and logarithmic above).

**Which pairs work best and worst** (PBMC, NB, 16D, mean over 3 seeds):
""")
code(r"""
q = pairs[(pairs.dataset == "pbmc") & (pairs.dim == 16) & (pairs.likelihood == "nb")].groupby(["protein", "gene"]).rank_pct.mean().sort_values()
best, worst = q.head(12).reset_index(), q.tail(12).reset_index()
display(pd.concat([best.rename(columns=lambda c: f"best: {c}"), worst.rename(columns=lambda c: f"worst: {c}")], axis=1).round(4))
""")
md(r"""
**Best**: classic lineage markers, CD16 (NK, monocytes), CD20 (B cells), CD127 (T cells), CD3, HLA-DR, CD123 (pDC). Their own gene is the nearest of ~20,000.

**Worst**: proteins that are almost absent in resting blood:
- activation markers: CTLA4, OX40/*TNFRSF4*, 4-1BB/*TNFRSF9*, GITR/*TNFRSF18*, PD-L1/L2;
- proteins atypical for blood: HER2/*ERBB2*, MCAM, ENG.

### Why: signal versus background
For each protein the model gives a **signal share**: the mean over cells of the fraction of the predicted count that comes from $e^{\rho_k - d}$ rather than from the background $e^{\beta_k}$.
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, ds in zip(axes, ["pbmc", "neurips"]):
    a = psig[psig.dataset == ds].groupby(["protein", "gene"]).mean(numeric_only=True).reset_index()
    rho, p = spearmanr(a.fg_share, a.rank_pct)
    ax.scatter(a.fg_share, a.rank_pct, s=12, alpha=0.7)
    for _, r in a[(a.rank_pct < 0.0005) | (a.rank_pct > 0.45)].iterrows():
        ax.annotate(r.protein, (r.fg_share, r.rank_pct), fontsize=6)
    ax.set_yscale("symlog", linthresh=0.001); ax.set_xlabel("signal share of protein count (1 − background)")
    ax.set_ylabel("percentile of own gene"); ax.set_title(f"{ds.upper()}: Spearman ρ = {rho:+.2f} (p = {p:.0e})", fontsize=8)
plt.tight_layout(); plt.show()
out = []
for ds in ["pbmc", "neurips"]:
    a = psig[psig.dataset == ds].groupby(["protein", "gene"]).mean(numeric_only=True)
    out.append({"dataset": ds, **{f"ρ with {n}": round(spearmanr(a[c], a.rank_pct)[0], 2) for c, n in
                [("fg_share", "signal share"), ("gene_mean", "gene expression"), ("protein_mean", "mean protein count")]}})
display(pd.DataFrame(out).set_index("dataset"))
""")
md(r"""
**Conclusion for H3: supported, strongly.**
- Median percentile of the own gene in 16D: **0.4–3%**.
- The larger the signal share of a protein's count, the closer it is to its gene. The relation is equally strong on both datasets (ρ ≈ −0.65).
- Gene expression explains this only on PBMC; the signal share explains it on both.

In other words, the model is **not wrong** on some proteins: proteins with no biological signal in the data simply have nothing to anchor to, and the model honestly attributes their counts to background.

**In 3D** the relation is weaker: median 10–14% on PBMC, 2–7% on NeurIPS. Three dimensions are not enough to place ~20k genes, 200 proteins and dozens of cell types without overlap.

### What it looks like in the joint space
PBMC, RNA + protein, NB, 16D. For the picture, the 16-dimensional space is reduced to 2D with UMAP, which preserves neighbourhoods. The UMAP is built on cells (6000 subsampled, coloured by `celltype.l1`), genes (3000 subsampled, grey) and marker proteins (black) together. Red points are the genes of these proteins; a line joins each pair. Below the picture: the exact rank of the own gene in the original 16-dimensional space, without UMAP distortion.
""")
code(r"""
emb = ad.read_h5ad("results/pbmc_full_rna+prot_nb_d16_s0/embedding.h5ad")
import re, umap
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
X = np.vstack([Z[ci], W[gi], V[pi_]])
E = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(X)
ec, eg, ep = E[:len(ci)], E[len(ci):len(ci) + len(gi)], E[len(ci) + len(gi):]
lab = emb.obs["celltype.l1"].astype(str).to_numpy()[ci]
fig, ax = plt.subplots(figsize=(8.5, 6.5))
ax.scatter(eg[:, 0], eg[:, 1], s=1, color="lightgrey", alpha=0.5, label="genes (subsample)")
for i, l in enumerate(sorted(set(lab))):
    m = lab == l; ax.scatter(ec[m, 0], ec[m, 1], s=2, alpha=0.4, label=l, color=plt.cm.tab10(i))
pair_rank = {}
for k, (p_, g) in enumerate(show_pairs):
    gv = eg[list(gi).index(genes.index(g))]; pv = ep[k]
    ax.plot([pv[0], gv[0]], [pv[1], gv[1]], color="k", lw=0.7)
    ax.scatter(*pv, color="k", s=22, zorder=5); ax.scatter(*gv, color="red", s=16, zorder=5)
    ax.annotate(p_, pv, fontsize=7, fontweight="bold"); ax.annotate(g, gv, fontsize=6, color="red", xytext=(3, -8), textcoords="offset points")
    d = np.linalg.norm(W - V[pnames.index(p_)], axis=1); pair_rank[f"{p_} → {g}"] = int((d < d[genes.index(g)]).sum()) + 1
ax.legend(frameon=False, fontsize=7, markerscale=4, loc="upper right"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("UMAP of the joint space: cells, genes (grey), proteins (black) and their genes (red)", fontsize=9); plt.show()
display(Markdown("**Rank of the own gene among all ~20,700 genes by distance to the protein (16D, 1 = nearest):** " +
                 ", ".join(f"{k}: **{v}**" for k, v in pair_rank.items())))

""")
md(r"""
How to read the picture:
- **Proteins** (black) sit **inside their own cell clusters**: CD19, CD20 and CD1c in B cells, CD14 in monocytes, CD16 and CD56 in NK cells, CD3, CD4 and CD127 in T cells.
- **Their genes** (red) stay in the gene cloud, on its edge facing the same cell type.

So a protein is closer to cells than to genes. This is expected: protein counts are dense and large and strongly determine where a cell goes. Still, **among genes** the own gene is one of the nearest to the protein:
- for CD16, CD127 and CD20 it is first or second of ~20,700;
- for HLA-DR, CD123, CD3 it is in the top ten;
- for CD4, CD8 and CD56 it is worse (rank 250–1100, i.e. top 1.5–5%).

UMAP preserves neighbourhoods only approximately, so line lengths are only an illustration. The exact test is the ranks computed in 16D.
""")

# ---------------------------------------------------------------- H4 / H5
md(r"""
## 5. H4 and H5 — New donors (leave-one-donor-out, PBMC)

*Hypotheses:*
- **H4:** fold-in transfers labels to a new donor almost as well as the reference, and better than aggregation;
- **H5:** totalVI is a strong baseline; multimodal SCENE will be comparable, not better.

For each of the 8 donors the model is trained on the other seven (donor is a batch level). The held-out donor's cells are then embedded one way or another, and cell types are transferred from the reference by kNN (k = 15). scVI and totalVI use the standard scvi-tools route for new data, **scArches**: the reference weights are frozen and only the parameters for the new data are trained. Mean ± standard deviation over 8 donors.
""")
code(r"""
def lodo_table(label):
    d = lodo[lodo.label == label].copy()
    d["embedding"] = d.embedding.map({"aggregate": "aggregation", "foldin": "fold-in", "scArches": "scArches"})
    d["model"] = d.model.str.replace("SCENE rna+prot", "SCENE RNA+protein").str.replace("SCENE rna", "SCENE RNA") \
        .str.replace(" zip", " ZIP").str.replace(" nb", " NB").replace({"scvi": "scVI (RNA)", "totalvi": "totalVI (RNA+protein)"})
    return d.groupby(["model", "embedding"]).agg(accuracy=("accuracy", ms), macro_F1=("macro_f1", ms), donors=("accuracy", "size"))
display(Markdown("**Fine types (`celltype.l2`, 30 types)**")); display(lodo_table("celltype.l2"))
display(Markdown("**Coarse types (`celltype.l1`, 8 types)**")); display(lodo_table("celltype.l1"))
""")
code(r"""
d = lodo[lodo.label == "celltype.l2"].copy()
d["key"] = d.model + " / " + d.embedding
keys = ["SCENE rna zip / aggregate", "SCENE rna zip / foldin", "SCENE rna nb / foldin", "SCENE rna+prot zip / foldin",
        "SCENE rna+prot nb / aggregate", "SCENE rna+prot nb / foldin", "scvi / scArches", "totalvi / scArches"]
names = ["SCENE RNA ZIP, aggregation", "SCENE RNA ZIP, fold-in", "SCENE RNA NB, fold-in", "SCENE RNA+protein ZIP, fold-in",
         "SCENE RNA+protein NB, aggregation", "SCENE RNA+protein NB, fold-in", "scVI, scArches", "totalVI, scArches"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, metric, title in [(axes[0], "accuracy", "accuracy"), (axes[1], "macro_f1", "macro-F1")]:
    for i, k in enumerate(keys):
        v = d[d.key == k].sort_values("donor")[metric].to_numpy()
        ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=12, alpha=0.7)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color="k", lw=1.5)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax.set_title(f"Fine-type transfer to the held-out donor: {title}", fontsize=8)
plt.tight_layout(); plt.show()
""")
md(r"""
Dots are the 8 held-out donors, the black bar is the mean.

**Conclusion for H4: supported.**
- Fold-in is much better than aggregation: on fine types 0.83–0.90 vs 0.48–0.53.
- **Proteins clearly help fold-in:** 0.83 → **0.90** accuracy, 0.63 → **0.73** macro-F1.
- Aggregation without training is weak on 30 fine types: averaging gene positions pulls cells toward the centre of the space and fine differences are lost. It works better for the 8 coarse types.
- Fold-in of one donor (~4k cells, 200 epochs) takes seconds.

**Conclusion for H5: supported.**
- totalVI stays best: 0.914 / 0.785. Multimodal SCENE is close: 0.896 / 0.726. Both beat scVI (0.879 / 0.712).
- SCENE is level with the best method on accuracy and slightly behind on macro-F1, i.e. on rare types.
- Its advantage is **interpretability**: genes and proteins share the space with the cells (section 4). totalVI and scVI have nothing like that.
""")

# ---------------------------------------------------------------- 3D vs 16D, summary
md(r"""
## 6. 3D versus 16D
""")
code(r"""
rows = []
for ds, lab in LAB.items():
    for dim in [3, 16]:
        d = full[(full.dataset == ds) & (full.dim == dim) & (full.likelihood == "nb")]
        rows.append({"dataset": ds, "dimension": f"{dim}D",
                     "NMI (RNA)": d[d.modality == "rna"][f"{lab}_nmi"].mean(), "NMI (RNA+protein)": d[d.modality == "rna+prot"][f"{lab}_nmi"].mean(),
                     "AUC": d.rna_auc_roc.mean(), "NLL": d.rna_heldout_nll.mean(),
                     "protein↔gene: median": d[d.modality == "rna+prot"].pg_median_rank.mean(),
                     "protein↔gene: in top 5%": d[d.modality == "rna+prot"].pg_top5.mean()})
pd.DataFrame(rows).set_index(["dataset", "dimension"]).round(3)
""")
md(r"""
3D is worse everywhere, but not catastrophically:
- AUC loses ~0.005;
- NMI loses 0.05–0.10;
- the protein ↔ gene match suffers most.

This agrees with the reproduction: a very low dimension is enough to predict detection, but fine structure (cell types, protein–gene pairs) needs 8–16 dimensions. **3D is good for pictures, 16D for analysis.**

## 7. Follow-up experiments

### Protein weight in the loss
The loss is $-\sum\log P(\text{RNA}) - w\sum\log P(\text{protein})$. PBMC, NB, 16D; NMI over 3 seeds, new-donor transfer on donors P1, P3, P5, P7.

| Protein weight $w$ | NMI (30 types) | New donor, accuracy / macro-F1 | Protein↔gene, median | RNA NLL |
|---|---|---|---|---|
| 0.1 | 0.626 | 0.850 / 0.634 | 3.8% | 1.173 |
| 0.3 | 0.626 | 0.868 / 0.663 | 3.9% | 1.173 |
| 1 (default) | 0.679 | 0.898 / 0.729 | 3.1% | 1.174 |
| **3** | **0.702** | **0.918 / 0.779** | 3.0% | 1.176 |
| 10 | 0.694 | 0.914 / 0.770 | 2.4% | 1.180 |

More protein weight improves cell types and transfer at a small cost in RNA fit. At $w = 3$ SCENE ties totalVI on the same four donors (0.916 / 0.777) and nearly matches its NMI (0.707). Caveat: $w$ was chosen on the same data, without a separate validation split.

### Isotype controls
Isotype controls are antibodies that bind nothing, so their whole count is background. The model gives them a signal share of 0.003 (Rat-IgG2b), 0.005 (Rat-IgG1-2) and 0.007 (Rag-IgG2c), and 0.10 for Rat-IgG1-1, against a median of 0.42 for real proteins. The background term does what it was designed for.

### Predicting the proteins of a new donor from RNA only
The held-out donor's cells are placed using RNA only, then each protein is predicted per cell; the table gives the mean Pearson correlation over proteins, over 8 donors.

| Method | Pearson r |
|---|---|
| totalVI (neural decoder, reference batches) | **0.531 ± 0.019** |
| kNN on RNA (PCA of 3000 variable genes, k = 15) | 0.492 ± 0.017 |
| SCENE positions + kNN | 0.474 ± 0.015 |
| SCENE, protein read from distance (NB) | 0.434 ± 0.030 |
| SCENE, protein read from distance (ZIP) | 0.414 ± 0.023 |

totalVI beats kNN for all 8 donors, and kNN beats SCENE for all 8. **A negative result for SCENE**: each protein is read from one distance to one point, which cannot describe a protein that is high in two different cell types. SCENE positions + kNN is close to kNN on raw RNA, so the cell map itself is fine; the loss comes from the readout.

## 8. Hypotheses: summary

| | Hypothesis (written before results) | Result |
|---|---|---|
| H1 | NB beats ZIP on held-out likelihood; AUC similar; decoupled slightly better AUC | ✅ NB better on NLL, AUC the same. ❌ decoupled gave nothing. NB's worse deviance is an artefact of the metric, not of the model |
| H2 | proteins improve clustering of fine types | ✅ moderately (+0.01–0.06 NMI), for all likelihoods on both datasets |
| H3 | a protein is closer to its own gene than a random gene | ✅ strongly: median 0.4–3%. Failures are explained by lack of signal (ρ ≈ −0.65) |
| H4 | fold-in beats aggregation and is close to the reference | ✅ 0.90 vs 0.48 (fine types); proteins help fold-in (+0.07) |
| H5 | totalVI is strong; SCENE comparable, not better | ✅ 0.896 vs 0.914 accuracy (tie at protein weight 3); SCENE beats scVI |

## 9. Limitations
- **Subsets:** 30k cells per dataset; LODO only on PBMC, with one seed per donor.
- **Clustering by KMeans** with $k$ = number of types is a coarse metric, sensitive to type imbalance.
- **Baselines** (scVI, totalVI) with default settings (16D, 400 epochs), no hyperparameter search. SCENE was not tuned either; we used the authors' settings.
- **The ADT → gene dictionary** is hand-made (~110 unambiguous pairs per dataset). Proteins without an unambiguous gene, such as isotype controls and TCR segments, are not in the H3 test.
- **The H3 test** compares with all genes, not with genes of matched expression.
- **No ablation** of the protein background term.

## 10. How to reproduce
```
python scripts/prepare_data.py all                   # data
python scripts/make_grid.py cuda                     # grid of 132 runs
sbatch --export=ALL,WORKER=0 scripts/run_grid.sbatch # and WORKER=1 (two GPU workstations)
python scripts/collect_results.py                    # tables
python scripts/analyze_likelihood.py mps             # ZIP vs NB calibration
python scripts/analyze_proteins.py mps               # protein signal
python scripts/build_trackA_report.py                # this report
```
All embeddings are in `results/<run_id>/embedding.h5ad`: cells in `obsm["X_scene"]`, genes in `varm["W_rna"]`, proteins in `uns["protein_positions"]`.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(PROJ)}}).execute()
nbf.write(nb, NB)
subprocess.run([str(Path(sys.executable).parent / "jupyter"), "nbconvert", "--to", "html", "--no-input", "--output", str(HTML), str(NB)], check=True)
print(f"wrote {NB.name}, {HTML.name} ({HTML.stat().st_size / 1e6:.1f} MB)")
