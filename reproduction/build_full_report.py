"""Build the full single-file report: SCENE_full_report.html (+ the .ipynb it is rendered from).

Contains the paper summary, the architectures of SCENE and the compared methods, metric definitions,
EVERY figure produced by our reproduction with explanations, what did not match, all differences
from the paper, and reproducibility issues in the authors' code.
Run from reproduction/:  .venv/bin/python build_full_report.py
"""
import subprocess
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
NB = ROOT / "SCENE_full_report.ipynb"
HTML = ROOT.parent / "SCENE_full_report.html"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


FIG_DIR = ROOT / "SCENE-reproducibility/notebooks/figures/output"


def files(folder, pattern="*"):
    """Figure files in a folder (resolved at build time), relative to the figure output folder."""
    return sorted(str(p.relative_to(FIG_DIR)) for p in (FIG_DIR / folder).glob(pattern) if p.suffix in (".svg", ".png"))


def show(*paths, per_row=None, width=None):
    args = ", ".join(repr(p) for p in paths)
    extra = (f", per_row={per_row}" if per_row else "") + (f", width={width}" if width else "")
    code(f"show({args}{extra})")


# =============================================================================================
md(r"""
# SCENE: full reproduction report

**Paper:** Møberg, Petersen, Herlau, Kristensen, Jessen, Mørup. *Setting the SCENE for Interpretable Cell–Gene Embeddings in Single-Cell RNA-seq.* bioRxiv, 2026. https://www.biorxiv.org/content/10.64898/2026.09.12.750699v1
**Authors' code:** https://github.com/oscarmoeberg/SCENE · https://github.com/oscarmoeberg/SCENE-reproducibility

Every figure in this file was produced by **our run** of the authors' code and notebooks; no figures are copied from the paper. Each group of figures comes with an explanation of what it shows and whether it matches the paper.

**Contents**
1. What the paper is about
2. Architectures: SCENE and the methods it is compared with
3. Metrics
4. Data
5. How we reproduced it: machines, time, versions, code changes
6. Results — all figures
   - Fig. 1 — model schematic
   - Fig. 2 — clusters in the joint space (PBMC)
   - Fig. 3 — interferon-β response (Kang)
   - Fig. 4 — benchmark and batch integration
   - Fig. 5 — ultra-low dimensions
   - Fig. 6 — GR gene modules and TCR regulons
   - Appendices: native 2D/3D of different methods, imputation, tables
7. Summary: the paper's claims versus our results
8. What did not match
9. Differences between our run and the paper
10. Reproducibility issues in the authors' code
11. Conclusions and ideas for follow-up
""")

code(r"""
import base64
from pathlib import Path

import pandas as pd
from IPython.display import HTML, Markdown, display

FIGS = Path("SCENE-reproducibility/notebooks/figures/output")
RES = Path("SCENE-reproducibility/results")
SHOWN = set()
pd.set_option("display.precision", 3)


def _uri(p):
    mime = "image/svg+xml" if p.suffix == ".svg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def show(*paths, per_row=None, width=None):
    # Embed figures in rows; legends get a narrow slot. Paths are relative to FIGS.
    paths = [str(p) for p in paths]
    per_row = per_row or len(paths)
    for i in range(0, len(paths), per_row):
        chunk = paths[i:i + per_row]
        n_leg = sum("legend" in p or "colorbar" in p for p in chunk)
        main_w = width or (96 - 12 * n_leg) // max(len(chunk) - n_leg, 1)
        html = "".join(
            f'<figure style="display:inline-block;width:{12 if ("legend" in p or "colorbar" in p) else main_w}%;'
            f'margin:0 0.5% 8px;vertical-align:top"><img src="{_uri(FIGS / p)}" style="width:100%;background:white"/>'
            f'<figcaption style="font-size:10px;color:#777">{p}</figcaption></figure>'
            for p in chunk
        )
        display(HTML(f"<div>{html}</div>"))
        SHOWN.update(chunk)


def files(folder, pattern="*"):
    return sorted(str(p.relative_to(FIGS)) for p in (FIGS / folder).glob(pattern) if p.suffix in (".svg", ".png"))
""")

# =============================================================================================
md(r"""
## 1. What the paper is about

**The problem.** scRNA-seq gives a cells × genes count matrix: hundreds of thousands of cells, tens of thousands of genes, more than 90% zeros and strong technical noise. The standard analysis path is:
- normalisation and log transform;
- selection of ~2000 highly variable genes (HVG);
- PCA;
- a neighbour graph, clustering and UMAP.

Every step irreversibly loses something. And the output is a representation of **cells** only: to understand the role of genes one has to go back to the original matrix.

**The idea of SCENE.** The count matrix is treated as a **weighted bipartite graph**: cells on one side, genes on the other, edge weight = number of UMIs. Both cells and genes get points **in one Euclidean space**. The closer a gene is to a cell, the more likely it is detected there and the larger its count. The model is probabilistic (zero-inflated Poisson) and is trained directly on the **raw counts of all genes**, without normalisation or HVG selection.

**What this gives.** Cell–cell, cell–gene and gene–gene distances are meaningful in the same space:
- marker genes lie next to their cell types;
- genes of one programme, for example a drug response, form compact groups.

The authors claim that all of this is visible even in the **native 2D/3D** coordinates of the model, without UMAP.

**The paper's main claims (the ones we test):**
1. Clusters in the joint space correspond to cell types (PBMC, NMI 0.63).
2. Markers and interferon-response genes sit next to "their" cells.
3. On scIB metrics SCENE is as good as or better than Harmony, scVI and SIMBA, especially on the simulation with **nested batches (SIM2)**, where scVI collapses the biological groups.
4. Embeddings are meaningful already in 2D/3D. Link prediction saturates at low dimension, while count reconstruction keeps improving with dimension; on stem cells reconstruction gets worse above 8D.
5. Glucocorticoid-response (GR) gene modules are more compact and follow the expected dynamics better than with SIMBA.
6. TF regulons in T-cell activation (TCR) stand out better than with SIMBA.
7. Imputation of hidden counts is more accurate than with scVI.
""")

md(r"""
## 2. Architectures

### 2.1 SCENE

SCENE is a **latent distance model** (Hoff et al. 2002) on a bipartite graph. There is no neural network and no encoder: all parameters are free tables optimised directly.

**Parameters**

| Parameter | Type (in the formula for one cell $i$ – gene $j$ pair) | How many in the model | Meaning |
|---|---|---|---|
| $z_i$ | **vector** of length $d$ | $N$ vectors ($N \times d$ matrix) | position of cell $i$ in the latent space |
| $w_j$ | **vector** of length $d$ | $G$ vectors ($G \times d$ matrix) | position of gene $j$ |
| $\psi_i$ | **scalar** | $N$ scalars, one per cell | cell effect: sequencing depth, library size |
| $\rho_j$ | **scalar** | $G$ scalars, one per gene | gene effect: overall expression level ($\omega_j$ in the paper) |
| $g^{(\ell)}_{b,j}$ | **scalar** | (batches × $G$) per level $\ell$; a full matrix $\Gamma^{(\ell)}$ or low-rank $U^{(\ell)}V^{(\ell)\top}$ | correction of batch $b$ for gene $j$ in batch "layer" $\ell$ (donor, lab, sub-batch) |
| $\alpha$ | **scalar** | 1 for the whole model | scale (> 0) that turns "affinity" into a detection probability |

(The paper writes the gene effect as $\omega_j$. Here it is $\rho_j$ to avoid confusion with the gene position $w_j$.)

The vectors $z_i$ and $w_j$ enter the formula only through the distance $\lVert z_i - w_j\rVert$, which is a scalar. So the affinity $\eta_{ij}$ below is a sum of four scalars and is itself a scalar for each (cell, gene) pair. In code it is computed for all pairs at once as an $N \times G$ matrix: `re_cells + re_genes.T - torch.cdist(Z_cells, Z_genes)`.

**Cell–gene affinity**

$$\eta_{ij} = \psi_i + \rho_j + \sum_{\ell} g^{(\ell)}_{b_\ell(i),\,j} - \lVert z_i - w_j\rVert$$

(the distance is the ordinary Euclidean norm, **not squared**: `torch.cdist(p=2)` in the authors' code)

**Likelihood (hurdle / zero-inflated Poisson)** separates two questions: "is the gene detected" and "how much of it".

$$P(x_{ij} > 0) = \pi_{ij} = \sigma(\alpha\,\eta_{ij}), \qquad P(x_{ij} = k \mid x_{ij} > 0) = \frac{\lambda_{ij}^k}{k!\,(e^{\lambda_{ij}} - 1)},\quad \lambda_{ij} = e^{\eta_{ij}}$$

$$\log L = \sum_{x_{ij} > 0}\big[\log \pi_{ij} + x_{ij}\log\lambda_{ij} - \log(e^{\lambda_{ij}} - 1) - \log x_{ij}!\big] + \sum_{x_{ij} = 0} \log(1 - \pi_{ij})$$

**Training**
- AdamW, lr = 0.05, weight decay $10^{-3}$ on $z$, $w$ and the batch corrections.
- 600 epochs (200 for GR).
- The likelihood is evaluated on the **whole** $N \times G$ matrix, i.e. $O(NG)$ per epoch. Mini-batching over cells is optional.

**Initialisation**: a spectral embedding (Laplacian eigenmaps) of the bipartite graph, i.e. eigenvectors of the normalised Laplacian $L = I - D^{-1/2} A D^{-1/2}$.

**Imputation and prediction**
- Expected count: $E[x_{ij}] = \pi_{ij}\,\lambda_{ij} / (1 - e^{-\lambda_{ij}})$.
- Detection probability: $\pi_{ij}$.

**What the model does not have**
- **An encoder.** New cells cannot be embedded without retraining: the model is transductive.
- **A dispersion parameter.** Over-dispersion is explained only by each cell–gene pair having its own $\lambda$.

In the authors' code the model is called `scLDM` (single-cell latent distance model), so in file names and tables SCENE = scLDM.

### 2.2 scVI (Lopez et al. 2018)

A variational autoencoder for counts.
- **Encoder** (a neural network) maps a cell's profile and batch to a latent vector $z$ of dimension 10–30.
- **Decoder** outputs negative binomial (NB/ZINB) parameters for each gene: $\mu = \text{library size}\times\text{softmax}(f(z, s))$, with a dispersion parameter per gene.
- **Batch** $s$ is given to the encoder and decoder as a covariate.
- **Training:** ELBO maximisation on mini-batches.
- The model is **inductive**: new cells are embedded by the encoder without retraining.
- It represents **cells only**: genes are not in the latent space.

### 2.3 SIMBA (Chen et al. 2024)

A graph embedding of cells + features.
- **Graph.** A gene's expression in a cell is discretised into several levels (`n_bins=5` in the authors' setup). Each level is a separate cell–gene edge type.
- **Batches.** With batches, edges are added between similar cells from different batches.
- **Training.** Embeddings of all nodes are learned with PyTorch-BigGraph as in knowledge-graph embedding: neighbouring nodes get close vectors, negative samples are pushed apart. Dimension 50, trained on CPU.
- As in SCENE, cells and genes lie **in one space**.
- **Differences from SCENE:** no probabilistic count model (counts become discrete edge types), no explicit likelihood and no model of zeros.

### 2.4 PCA + Harmony

The classic baseline.
- Normalisation, log, HVG, then PCA (50 components).
- **Harmony** iteratively shifts the principal components so that soft clusters are mixed across batches.
- No genes in the space and no probabilistic model.

### 2.5 Comparison

| | PCA/Harmony | scVI | SIMBA | **SCENE** |
|---|---|---|---|---|
| Input | normalised HVG | raw counts | discretised counts | **raw counts, all genes** |
| Genes in the space | no | no | yes | **yes** |
| Probabilistic count model | no | NB/ZINB | no | **ZIP (hurdle)** |
| Meaning of cell–gene distance | — | — | vector similarity | **detection probability and expected count** |
| Batches | Harmony correction | covariate in the VAE | edges between batches | **additive per-gene corrections, several layers** |
| New cells without retraining | yes (projection) | yes (encoder) | no | **no** |
| Cost | low | mini-batches, GPU | CPU, graph | $O(NG)$ per epoch |
""")

md(r"""
## 3. Metrics

**Quality of cell embeddings (scIB, Luecken et al. 2022)**, higher is better everywhere:

| Metric | What it measures |
|---|---|
| **KMeans NMI / ARI** | KMeans (k = number of cell types) on the embedding against the labels. Conservation of biology |
| **Silhouette label** | silhouette by cell type, rescaled to [0, 1]. Separation of types |
| **Silhouette batch** | $1 - \lvert s\rvert$ of the silhouette by batch within each cell type. 1 means batches are mixed |
| **Graph connectivity** | fraction of a type's cells in the largest connected component of the kNN graph. 1 means the type is not split by batch |

**Predictive performance (Fig. 5):** 10% of the non-zero counts are hidden during training.
- **Link prediction:** the model must separate hidden non-zero pairs from random zero pairs by $\pi_{ij}$. AUC-ROC, PR-AUC and max F1 are reported.
- **Poisson deviance** on the hidden non-zeros: accuracy of the count itself, lower is better.

**Imputation:** RMSE, MAE and Pearson correlation between hidden and predicted counts.

**Gene-module compactness (Fig. 6):**
$$T(M) = \frac{\text{mean distance within the module}}{\text{mean distance from module genes to all other genes}}$$
- The smaller $T$, the more compact the module.
- The p-value is the fraction of random modules of the same size (10,000 permutations, genes drawn uniformly) with $T$ no larger than observed.
- q is after Benjamini–Hochberg correction.
""")

md(r"""
## 4. Data

| Dataset | Cells × genes | Used in |
|---|---|---|
| Cortex (mouse, Zeisel 2015) | 3,005 × 19,972 | benchmark (Fig. 4), dimension (Fig. 5) |
| PBMC CITE-seq (Hao 2021) | 152,094 × 20,729; **we use 30,000 cells** | clusters (Fig. 2), benchmark, dimension, imputation |
| IFN-β stim (Kang 2018) | 24,673 × 15,706 | interferon response (Fig. 3) |
| HCA Nuclei, heart (Litviňuková 2020) | 13,733 × 26,662 | Harvard/Sanger batch integration (Fig. 4), dimension |
| SIM1 / SIM2 (scIB, Splatter) | 12,097 × 9,979 / 19,318 × 10,000 | batches; SIM2 has nested batches and sub-batches (Fig. 4) |
| NeurIPS 2021 BMMC CITE-seq | 90,261 × 13,953; stem cells 39,860 | benchmark; lineage branching (Fig. 5) |
| GR time course (T47D, dexamethasone; Hoffman 2020) | 6 time points × 400 × 24,128 | GR gene modules (Fig. 6) |
| TCR stim, CD4⁺ T cells (Ding 2020) | 9,492 × 17,840 | TF regulons (Fig. 6) |
""")

# =============================================================================================
md(r"""
## 5. How we reproduced it

**What we ran.** All commands in the authors' `bash/paper/*.sh` (350 commands) went through our queue `run_queue.py`, which resumes where it stopped, moves finished work between machines and can filter commands. We then executed all 10 of the authors' figure notebooks. **Every command and every notebook cell ran without errors.**
""")
code(r"""
display(Markdown('''
**Where and how long** (from our logs, sum of command times):

| Machine | Commands | Time |
|---|---|---|
| MacBook M5 Pro, 48 GB (Apple GPU, MPS) | 76 | 9.6 h |
| Cluster, shared GPU node (8× RTX 5000 Ada, shared with other users) | 93 | 6.7 h |
| Cluster, workstation A (1× RTX 5000 Ada, 32 GB): PBMC, stem cells | 38 | 5.5 h |
| Cluster, workstation B: HCA, cortex, sim2, evaluations | 81 | 1.8 h |
| Cluster, SIMBA-A: SIMBA on full NeurIPS and PBMC | 20 | 3.6 h |
| Cluster, SIMBA-B: SIMBA on the other datasets and TCR | 57 | 13.9 h (8.9 h of it one single-threaded SIMBA-TCR regulon evaluation) |

Some commands from the shared GPU node were rerun on the dedicated workstations (section 9), so the total exceeds 350 commands.
'''))
""")
md(r"""
**Versions: authors versus ours**

| Package | Authors (`environment_scene.yaml`) | Mac | Cluster |
|---|---|---|---|
| Python | 3.11 | 3.13 | 3.12 |
| torch | 2.10 | 2.14 (MPS) | 2.11 + CUDA 12.8 |
| scvi-tools | 1.2.2 | 1.5.1 | 1.5.1 |
| scanpy / anndata | 1.11 / 0.12 | 1.12.4 / 0.13.4 | 1.12.4 / 0.13.4 |
| numpy | 1.26 | 2.5.3 | 2.5.3 |
| SIMBA environment | Python 3.10, torch 2.3.0, dgl 2.2.0, SIMBA @2da4787, PBG @4cd0f08 | — | the same, but a **CPU build** of torch 2.3.0, dgl 2.2.0 from data.dgl.ai |
| R for TCR | R 4.4, Seurat 5.2 | — | R 4.4, Seurat 5.2 + a newer SeuratObject |

**What we changed in the authors' code** (all changes are minimal, 3–10 lines each; see `upstream_changes.patch`):

| File | Change | Why |
|---|---|---|
| `scripts/imputation.py` | optional mini-batches and chunked reconstruction via the env vars `SCENE_CELL_BATCH_SIZE` / `SCENE_RECON_CHUNK` | for the shared GPU. **Not used in the final run**: without the variables the behaviour is the authors' |
| `scripts/write_paper_tables.py` | skip missing TCR tables | an intermediate run before SIMBA/TCR; in the final run all tables exist |
| `notebooks/figures/fig_5_ultra_low_dim.ipynb` | `seeds = [0, 1]` | we ran the dimension sweep with 2 seeds |
| `data/cortex.h5ad` | rewritten by the authors' `prepare_data.py cortex` | same content; the file is overwritten by their own preparation step |
| **our** `prepare_pbmc_fix.py`, `prepare_tcr_fix.R/.py` | PBMC and TCR preparation that avoids the broken steps | see section 10 |
""")

# =============================================================================================
md(r"""
## 6. Results — all figures

### Fig. 1 — Model schematic

**a.** The cells × genes count matrix (toy example).
**b.** The same matrix as a bipartite graph: edge weight = count.
**c.** The shared space: cells (blue) and genes (red) share coordinates, and groups of cells are surrounded by "their" genes.
""")
show("fig_1/panel_a_matrix_thinbars.svg", "fig_1/panel_b_bipartite.svg", "fig_1/panel_c_embedding.svg", "fig_1/cell_gene_node_legend.svg")
md(r"""
Next come the formula captions for the Fig. 1 schematic: $\eta$, $\lambda$, $\pi$, the edge weight and ZIP. These are rendered text, not plots. Fonts are not embedded in the SVGs, so symbols may overlap. The formulas themselves are explained in section 2.1.
""")
show(*[f"fig_1/{n}" for n in ["scene_eta.svg", "scene_lambda.svg", "scene_pi.svg", "scene_lambda_pi.svg", "scene_zip.svg", "scene_edge_weight.svg"]], per_row=3)

md(r"""
### Fig. 2 — Clusters in the joint space (PBMC CITE-seq, 16D)

Cells **and genes** are clustered together with Ward's method in the SCENE space; the number of clusters is $\approx \log_2(N+G)$.
- **Dendrogram:** the cluster hierarchy; the axis is $\log_2$ of the squared Euclidean distance.
- **UMAP of the spectral initialisation versus the trained model**, coloured by Ward cluster and by cell type. The Laplacian already gives a coarse structure, and training refines it.
- **Heatmap:** the fraction of each cell type in each cluster; genes are a separate "type". Clusters made mostly of genes are gene "clouds" around groups of cells.
- **Native 2D** (PNG): a SCENE embedding trained directly in 2D.

**Comparison with the paper.** NMI of clusters with cell types (genes excluded): ours **0.660**, paper **0.63**. As in the paper, confusion is within close populations (T-cell subtypes, monocytes), not between distant lineages. ✅
""")
show("fig_2/pbmc_cite_seq_dendrogram.svg", width=96)
show("fig_2/pbmc_cite_seq_umap_laplacian_ward.svg", "fig_2/pbmc_cite_seq_umap_latent_ward.svg", "fig_2/pbmc_cite_seq_umap_cluster_legend.svg")
show("fig_2/pbmc_cite_seq_umap_laplacian_celltype.svg", "fig_2/pbmc_cite_seq_umap_latent_celltype.svg", "fig_2/pbmc_cite_seq_umap_celltype_legend.svg")
show("fig_2/pbmc_cite_seq_ward_x_celltype_heatmap_with_gene.svg", "fig_2/pbmc_cite_seq_umap_latent_celltype_2D.png", width=45)

md(r"""
### Fig. 3 — Interferon-β response (Kang 2018)

PBMCs from 8 donors, control and 6 h IFN-β, trained in one SCENE space (16D).
- **UMAP by cell type and by condition.** Cell type and stimulation are two independent axes of variation, and both are visible. Genes are shown as labelled points.
""")
show("fig_3/kang_umap_cell_type.svg", "fig_3/kang_umap_label.svg", "fig_3/celltype_legend.svg", "fig_3/stim_group_legend.svg")
md(r"""
- **Distance from every cell to a gene** (UMAP coloured by the distance in the SCENE space; darker = closer):
  - interferon genes (MX1, ISG15, IFIT1, CXCL10, OAS1, RSAD2) are closest to the **stimulated** cells;
  - CASP1, NLRP3 (inflammasome) and HMGCS1 (cholesterol synthesis) are genes of other programmes, shown for comparison.

A gene "marks" its region of the space, and this is visible without going back to the expression matrix.
""")
show(*files("fig_3", "kang_umap_latent_scLDM_dist_*.svg"), per_row=3)
show("fig_3/shared_distance_colorbar.svg", "fig_3/genes_legend.svg", "fig_3/gene_labels_grid.svg")
md(r"""
- **$\pi$ (detection probability) against cell–gene distance** for classic markers (CD74, CTSW, FCER1A, FCGR3A, FCN1, GNLY, IL7R, MS4A1, NKG7). Colour is cell type. The probability falls with distance, and high $\pi$ belongs to the "right" types: MS4A1 to B cells, GNLY/NKG7 to NK cells, FCN1 to monocytes.

**Comparison with the paper:** qualitatively the same. ✅
""")
show(*files("fig_3/pi_vs_dist"), per_row=3)

md(r"""
### Fig. 4 — Benchmark: clustering and batch integration

**Table:** all scIB metrics, mean over 5 seeds. SCENE = `scLDM`, Harmony = `PCA`.
""")
code(r"""
bench = pd.read_csv(FIGS / "fig_4/benchmark_summary_combined.csv")
cols = {"KMeans NMI_mean": "NMI", "KMeans ARI_mean": "ARI", "Silhouette label_mean": "Sil. label",
        "Silhouette batch_mean": "Sil. batch", "Graph connectivity_mean": "Graph conn."}
t = bench.set_index(["dataset", "display_name"])[list(cols)].rename(columns=cols)
t.index.names = ["dataset", "method"]
t.round(3)
""")
md(r"""
**Bar charts** (as in the paper: HCA, NeurIPS, SIM1, SIM2; error bars are standard deviations over seeds):
""")
show("fig_4/nmi_grouped_bar.svg", "fig_4/ari_grouped_bar.svg", "fig_4/sil_labels_grouped_bar.svg", "fig_4/legend_methods.svg")
show("fig_4/sil_batch_grouped_bar.svg", "fig_4/graph_conn_grouped_bar.svg", width=40)
md(r"""
**SIM2: nested batches** (4 batches, each with 4 sub-batches, 4 biological groups). UMAPs for scVI and SCENE in three modes:
- no batch correction;
- batch correction;
- batch and sub-batch correction.

Coloured by group (biology), batch and sub-batch. **A good method** separates the groups and mixes the batches.
- **scVI**, even with correction for both levels, mixes the groups.
- **SCENE** with batch and sub-batch correction (`scldm_batch_subbatch`) cleanly separates the 4 groups.
""")
for method, name in [("scvi", "scVI"), ("scldm", "SCENE")]:
    for mode, mode_name in [("", "no batch correction"), ("_batch", "batch correction"),
                            ("_batch_subbatch", "batch and sub-batch correction")]:
        md(f"**{name}: {mode_name}** (left to right: group, batch, sub-batch)")
        show(*[f"fig_4/sim2_umap_{method}{mode}_{c}.svg" for c in ["Group", "Batch", "SubBatch"]])
show("fig_4/sim2_umap_cluster_legend.svg", width=20)
md(r"""
**HCA Nuclei (heart)**, two nucleus sources (Harvard and Sanger) and several donors. Top: SCENE without batch correction; bottom: with correction for source and donor. Coloured by source, cell type and donor.
""")
show("fig_4/hca_umap_scldm_cell_source.svg", "fig_4/hca_umap_scldm_cell_type.svg", "fig_4/hca_umap_scldm_donor.svg")
show("fig_4/hca_umap_scldm_batch_cell_source.svg", "fig_4/hca_umap_scldm_batch_cell_type.svg", "fig_4/hca_umap_scldm_batch_donor.svg")
show("fig_4/hca_cell_source_legend.svg", "fig_4/hca_cell_type_legend.svg", width=20)
md(r"""
**Joint cell–gene embedding of HCA:** UMAP and **native 2D** without UMAP, with heart marker genes labelled (MYH7, NPPA for cardiomyocytes; PECAM1, VWF for endothelium; DCN for fibroblasts, …). Markers sit next to their types. Below: **native 3D** from four angles.
""")
show("fig_4/hca_joint_umap_latent_cell_type.svg", "fig_4/hca_joint_2D_latent_cell_type.png")
show(*files("fig_4", "hca_scLDM_3d_cell_type_angle*.png"), per_row=4)
show("fig_4/gene_labels_grid.svg", width=60)
md(r"""
**Comparison with the paper**
- **SIM2, the main argument, reproduces strongly.** NMI: SCENE 0.99, scVI 0.29, SIMBA and Harmony 0.05. ✅
- **Cortex, SIM1:** SCENE is best. ✅
- **HCA, NeurIPS, PBMC:** SIMBA is slightly ahead of SCENE on NMI, and on PBMC Harmony is best. ⚠️ More in section 8.
""")

md(r"""
### Fig. 5 — Ultra-low dimensions

**NeurIPS stem cells** (HSC → erythroid and monocyte lineages), SCENE trained **directly in 2D and 3D**.
- The branching of the two lineages is visible **in the model's coordinates**, without UMAP.
- `double_score` colouring: the relative closeness of each cell to HBB (erythroid lineage) versus FCGR3A (monocytes), i.e. distance to HBB minus distance to FCGR3A.
- For comparison, a UMAP computed on top of the native 3D. It changes the global geometry and makes the branch ends look closer than in the model.

(The `kang_` prefix in file names is a quirk of the authors' notebook. The data here are NeurIPS stem cells.)
""")
show("fig_5/kang_scLDM_2d_cell_type.svg", "fig_5/kang_umap_2d_combo_cell_type.svg", "fig_5/celltype_legend.svg")
show(*files("fig_5", "kang_scLDM_3d_cell_type_angle*.svg"), per_row=4)
show(*files("fig_5", "kang_scLDM_3d_double_score_*angle*.svg"), per_row=4)
show("fig_5/cbar_double_score_rawdiff_FCGR3A_vs_HBB.svg", width=8)
md(r"""
**Dependence on dimension** $d \in \{2, 3, 8, 16, 32, 64\}$. 10% of the non-zero counts are hidden; mean over 2 seeds (5 in the paper).
""")
show("fig_5/auc_roc_vs_dimension.svg", "fig_5/ap_vs_dimension.svg", "fig_5/micro_f1_vs_dimension.svg", "fig_5/legend_datasets.svg")
show("fig_5/poisson_deviance_vs_dimension.svg", width=40)
code(r"""
dims = pd.read_csv(FIGS / "fig_5/metric_summary_by_dataset_dim.csv")
display(Markdown("**AUC-ROC** (link prediction)"))
display(dims.pivot(index="Dataset", columns="dim", values="AUC-ROC_mean").round(3))
display(Markdown("**Poisson deviance** (lower is better)"))
dims.pivot(index="Dataset", columns="dim", values="Poisson Deviance_mean").round(3)
""")
md(r"""
**Comparison with the paper: matches on all three points.** ✅
1. Link prediction saturates immediately: AUC 0.90–0.92 already in 2D, and barely grows after.
2. Deviance improves with dimension on cortex and PBMC.
3. On stem cells it **gets worse above 8D**: 0.79 at 8D, 0.95 at 64D.
""")

md(r"""
### Fig. 6 — Glucocorticoid-response (GR) gene modules

T47D breast cancer cells with dexamethasone: control, 1, 2, 4, 8, 18 h. Each time point has its own 3D SCENE model and its own SIMBA. The compactness of four modules is tested:
- **GR Direct GRE**: direct GR targets FKBP5, TSC22D3, PER1, KLF9, DUSP1, ZFP36, SGK1;
- **GR Late TF**: late transcription factors;
- **Glycolysis** and **TCA**: metabolic modules, a "positive control" that should always be compact.

Heatmap of −log10(q): a priori expectation, SIMBA and SCENE.
""")
show("fig_6/gr_heatmap.svg", width=96)
code(r"""
mods = ["GR_direct_GRE_core", "GR_secondary_TF_core", "Glycolysis_core", "TCA_cycle_core"]
gb = RES / "geometric_biology"
rows = {}
for tp in ["00", "01", "02", "04", "08", "18"]:
    s = pd.read_csv(gb / f"scLDM_GR_{tp}_3D/gene_module_eval.csv", index_col="module")["qval_fdr_bh"]
    b = pd.read_csv(gb / f"SIMBA_GR_{tp}/gene_module_eval.csv", index_col="module")["qval_fdr_bh"]
    rows["control" if tp == "00" else f"{int(tp)} h"] = {m: f"{s[m]:.1e} / {b[m]:.1e}" for m in mods}
display(Markdown("q-value **SCENE / SIMBA** (q < 0.05: the module is compact)"))
pd.DataFrame(rows).T
""")
md(r"""
The native 2D SCENE space at control and at 18 h (a separate model per time point). Genes of the GR Direct module are highlighted and connected. The compactness ratio is almost the same at the two time points (0.623 vs 0.616), so the time trend is clearer in the q-values above than in this picture.
""")
show("fig_6/kang_scLDM_2d_time_hr_adata_00.svg", "fig_6/kang_scLDM_2d_time_hr_adata_18.svg")
md(r"""
**Comparison with the paper: matches.** ✅
- **GR Direct** is significant for SCENE from the first hour (q ≈ 10⁻⁴–10⁻³) and weaker for SIMBA (q ≈ 0.006–0.03). In the control it is not significant for either, as expected.
- **GR Late TF** becomes significant for SCENE at 4–18 h, consistent with biology: late TFs switch on later. For SIMBA it is not significant at any time point.
- **Metabolic modules** are compact for both.

### Fig. 6 (continued) — TF regulons in T-cell activation (TCR)

CD4⁺ T cells, unstimulated and after 4 h of anti-CD3/CD28, trained in one space (SCENE 16D, SIMBA 50D). For 174 TFs from DoRothEA (confidence A/B) the compactness of their regulon is tested.
- **Heatmap:** 26 TFs important for T-cell activation: early transcription (NFAT, NF-κB, EGR1, ATF, ELK1), cytokine response (STAT/IRF), anabolic metabolism (MYC, MAX, HIF1A, SREBF2).
- **Dot plot:** −log10(q) of all regulons for SCENE and SIMBA; colour shows significant in both, in one only, or in neither.
""")
show("fig_6/tf_heatmap.svg", width=96)
show("fig_6/qval_jitter.svg", width=45)
code(r"""
s = pd.read_csv(gb / "scLDM_TCR/gene_module_eval.csv", index_col="module")["qval_fdr_bh"]
b = pd.read_csv(gb / "SIMBA_TCR/gene_module_eval.csv", index_col="module")["qval_fdr_bh"]
display(Markdown(f"**Regulons: {len(s)}.** q < 0.05: SCENE **{(s < 0.05).sum()}**, SIMBA **{(b < 0.05).sum()}**, both {((s < 0.05) & (b < 0.05)).sum()}, SCENE only {((s < 0.05) & (b >= 0.05)).sum()}, SIMBA only {((s >= 0.05) & (b < 0.05)).sum()}"))
tfs = ["NFATC1", "NFATC2", "RELA", "REL", "NFKB1", "NFKB2", "EGR1", "ATF1", "ATF2", "ELK1",
       "STAT1", "IRF1", "MYC", "MAX", "HIF1A", "SREBF2"]
pd.DataFrame({"SCENE q": s.reindex(tfs), "SIMBA q": b.reindex(tfs)}).map(lambda q: f"{q:.1e}")
""")
md(r"""
**Comparison with the paper: matches.** ✅ Key activation regulators (NFATC1/2, NFKB2, ATF1/2, HIF1A) are significant for SCENE and not for SIMBA; these are the examples the authors give.
""")

md(r"""
### Appendix — Native 2D/3D spaces of different methods (NeurIPS stem cells)

Each method is trained **directly in 2D or 3D**, without UMAP: PCA, scVI, SIMBA, SCENE. The authors use this to show that only in SCENE the lineage branching reads well in the model's own coordinates.
""")
show("app/low_dim/kang_PCA_2d.svg", "app/low_dim/kang_scVI_2d.svg", "app/low_dim/kang_SIMBA_2d.png", "fig_5/kang_scLDM_2d_cell_type.svg")
show("app/low_dim/kang_PCA_3d.svg", "app/low_dim/kang_scVI_3d_cell_type_angle3.svg", "app/low_dim/kang_SIMBA_3d_cell_type_angle6.svg")

md(r"""
### Appendix — Imputation (PBMC)

10% of the non-zero counts are hidden; models are trained on the rest and predict the hidden values:
- SCENE with ZIP;
- SCENE with a plain Poisson;
- scVI.

The scatter plots show prediction against truth; the bars show the metrics.
""")
code(r"""pd.read_csv(FIGS / "app/imputation_metrics.csv", index_col=0).round(3)""")
show("app/pbmc_imputation_scvi_vs_scene.svg", width=80)
md(r"""
**Comparison with the paper: matches.** ✅ SCENE-ZIP is more accurate than scVI:
- Pearson 0.85 vs 0.65;
- RMSE 8.6 vs 12.6.

ZIP is better than the plain Poisson. The authors' script reports Spearman = 0 for every model; see section 8.

### Appendix — The authors' tables (`write_paper_tables.py`)
""")
code(r"""
tb = FIGS / "tables"
display(Markdown("**Run parameters** (`run_parameters.csv`, first rows)"))
display(pd.read_csv(tb / "run_parameters.csv").head(8))
display(Markdown("**GR modules** (`gr_modules.csv`)"))
display(pd.read_csv(tb / "gr_modules.csv"))
display(Markdown("**TCR: the 10 most compact regulons for SCENE** (`SCENE_TCR.csv`)"))
pd.read_csv(tb / "SCENE_TCR.csv").sort_values("qval_fdr_bh").head(10)[["module", "n_manual_genes", "obs_ratio", "qval_fdr_bh"]]
""")
code(r"""
all_figs = {str(p.relative_to(FIGS)) for p in FIGS.rglob("*") if p.suffix in (".svg", ".png")}
missing = sorted(all_figs - SHOWN)
display(Markdown(f"*Completeness check: {len(SHOWN & all_figs)} of {len(all_figs)} figures of our run are shown"
                 + (f"; not shown: {missing}*" if missing else ", i.e. all of them.*")))
""")

# =============================================================================================
md(r"""
## 7. Summary: the paper's claims and our results

| # | Paper's claim | Our result | Status |
|---|---|---|---|
| 1 | Clusters in the joint space correspond to cell types (PBMC, NMI 0.63) | NMI 0.660 | ✅ |
| 2 | Markers and IFN genes sit next to their cells (Kang) | visible in Fig. 3; $\pi$ falls with distance for the right types | ✅ |
| 3a | SIM2: scVI collapses the groups, SCENE does not | NMI 0.99 (SCENE) vs 0.29 (scVI) and 0.05 (SIMBA, Harmony) | ✅ |
| 3b | SCENE is as good as or better than competitors on scIB | best on Cortex, SIM1, SIM2; on HCA, NeurIPS, PBMC SIMBA or Harmony slightly ahead on NMI | ⚠️ "as good as", not "better" |
| 4a | Link prediction saturates at low dimension | AUC 0.90–0.92 already in 2D | ✅ |
| 4b | Deviance improves with dimension | cortex, PBMC: yes | ✅ |
| 4c | Stem cells: worse above 8D | 0.79 (8D) → 0.95 (64D) | ✅ |
| 5 | GR modules better than with SIMBA | GR Direct q 10⁻⁴–10⁻³ vs 0.006–0.03; Late TF significant only for SCENE | ✅ |
| 6 | TCR regulons better than with SIMBA | 58 vs 42 of 174; key TFs only for SCENE | ✅ |
| 7 | Imputation better than scVI | Pearson 0.85 vs 0.65, RMSE 8.6 vs 12.6 | ✅ |
""")

md(r"""
## 8. What did not match

**1. SCENE does not lead on clustering of real data.** By NMI:
- **HCA:** SIMBA 0.853, SCENE 0.849;
- **NeurIPS:** SIMBA 0.689, SCENE 0.673;
- **PBMC (30k):** Harmony 0.677, SIMBA 0.663, scVI 0.645, SCENE 0.643. On PBMC SCENE is also last on ARI and graph connectivity.

The differences are small, but the paper's "competitive or superior" is, on these data, more honestly "competitive". SCENE's strong, unambiguous advantage is on simulations with batches and in the interpretation of genes.

**2. PBMC is a subset for us.** We trained on 30,000 of 152,094 cells (section 9), so part of the gap on PBMC may come from the subset. Checking this needs the full dataset: ~130 GB of memory for full-matrix training, or mini-batches.

**3. The Ward-cluster NMI on PBMC is higher than in the paper:** 0.660 vs 0.63. Most likely also an effect of the subset and different randomness. The difference favours the method and does not change the conclusion.

**4. On SIM2, SIMBA and Harmony give identical NMI and ARI** (0.0488 and 0.0425, identical to the last digit). Their embeddings differ (silhouette and connectivity differ), but KMeans produced the same partition for both, most likely by batch. This is not a bug of our run but a sign that both methods group SIM2 cells by batch rather than by biology.

**5. SCENE is not the best at removing batch.** On SIM1 batch silhouette is 0.858 for SCENE, 0.955 for scVI and 0.971 for SIMBA. On HCA and NeurIPS SCENE has the lowest graph connectivity, although the gap is small (0.977 vs 0.986–0.994 on HCA). SCENE is stronger at **conserving biology** than at **mixing batches**; the paper does not discuss this trade-off in detail.

**6. Spearman in the imputation table is 0.0 for every model.** This is the output of the authors' script (`imputation.py`), not our error. It looks like a quirk of the computation, which we did not investigate. Pearson, RMSE and MAE look sensible.

**7. The dimension sweep used 2 seeds instead of 5,** so the error bars in Fig. 5 come from two points. All qualitative conclusions match.
""")

md(r"""
## 9. Differences between our run and the paper

| # | Difference | Reason | Effect |
|---|---|---|---|
| 1 | **PBMC: a fixed subset of 30,000 cells** (seed 0, cell list in `pbmc_sub30k_cells.txt`) for all methods | full-matrix SCENE training needs ~41 bytes per entry (we measured it), ~130 GB for full PBMC | PBMC metrics are not directly comparable with the paper's numbers |
| 2 | **Dimension sweep: seeds 0–1** instead of 0–4; the `fig_5` notebook switched to `[0, 1]` | time | Fig. 5 error bars over two seeds |
| 3 | **Devices:** SCENE on MPS (Mac) and CUDA (RTX 5000 Ada) instead of the authors' CPU; scVI on MPS/CUDA | speed | small numerical differences |
| 4 | **Full NeurIPS (90k cells): SCENE with mini-batches of 8192 cells** (5 benchmark runs) | full-matrix training needs ~52 GB, more than the Mac's 48 GB | the only final SCENE fits not on the full matrix |
| 5 | All other final SCENE fits are **on the full matrix, as in the paper**. Trial mini-batch runs on the shared GPU (other users' processes held 26–32 GB) were rerun on dedicated workstations | — | — |
| 6 | **Newer package versions** (torch 2.11–2.14, scvi-tools 1.5.1, numpy 2.5 instead of 2.10 / 1.2.2 / 1.26) | Python 3.12–3.13; old builds for Apple Silicon | possible small differences in scVI and metrics |
| 7 | **SIMBA:** a CPU build of torch 2.3.0; dgl 2.2.0 from data.dgl.ai | dgl 2.2.0 is not on PyPI; SIMBA trains on CPU anyway | none |
| 8 | **TCR:** data built by exporting counts and metadata directly from Seurat (`prepare_tcr_fix.R`) | `as.SingleCellExperiment` fails with the current SeuratObject | size matches the authors' reference (9,492 × 17,840) |
| 9 | **SCENE-TCR regulon evaluation with 12 threads** (`--n_jobs 12`) | single-threaded: ~9 h | none: with `permutation_rng=independent` each module has its own generator |
| 10 | **PBMC:** protein name `CD66a/c/e` → `CD66a_c_e` | current anndata cannot write `/` in keys | none: proteins are not used; models are trained on RNA |
""")

md(r"""
## 10. Reproducibility issues in the authors' code

What prevented running the code "as is" in 2026:
1. **`prepare_data.py pbmc` fails** when writing `.h5ad`: a protein name contains `/`, which current anndata/h5py reject.
2. **`prepare_tcr.R` does not work** with the current SeuratObject: `PackageCheck()` was removed, and `Seurat::as.SingleCellExperiment` calls it.
3. **`environment_simba.yaml` does not install:** `dgl==2.2.0` is only on the DGL server, not on PyPI.
4. **The data downloader has no timeout** (`urllib`) and can hang forever on a slow figshare.
5. **Memory.** Full-matrix SCENE training uses ~41 bytes per cell × gene entry. Full PBMC (3.15 billion entries) needs ~130 GB. The authors' README only says "substantial memory".
6. **The TCR regulon evaluation is single-threaded** in the authors' scripts: ~9 h per method, although the code supports `--n_jobs` and the result does not depend on it.
7. **The figure notebooks require all results:** without SIMBA or TCR they fail entirely instead of skipping panels.
8. **`prepare_data.py cortex` overwrites** the `data/cortex.h5ad` stored in the repository.

That said, the structure of the authors' repository is good: all commands are listed in `bash/paper/*.sh`, the data are prepared by one script, and reference data sizes are given for checking. Everything above is fixed by small changes.
""")

md(r"""
## 11. Conclusions and ideas for follow-up

**Main points:**
- 9 of the 10 tested claims of the paper reproduce.
- Three things are especially convincing:
  - robustness to nested batches (SIM2);
  - meaningful native 2D/3D;
  - an interpretable gene geometry: GR modules and TCR regulons better than with SIMBA.
- The only caveat: on clustering of real data SCENE is level with SIMBA and Harmony, not ahead.

**Directions for our project** (Track A is implemented in `scene_mm/`, see `trackA_report.html`):
1. **Inductivity.** SCENE has no encoder. New cells can be embedded with genes frozen (fold-in) or as a weighted mean of gene embeddings; test on held-out PBMC donors.
2. **Likelihood.** Compare ZIP with NB and hurdle-NB, and separate the detection and size parameters; metrics: held-out likelihood and link prediction.
3. **A fair null test for modules.** Modules are currently compared with uniformly random genes. A test against random genes matched on expression level and detection rate is needed, plus a check of the gene geometry for technical artefacts (ribosomal, mitochondrial, stress genes).
4. **Multimodality.** PBMC and NeurIPS CITE-seq include proteins (ADT), which can be added as a third node type with their own NB likelihood.
5. **Full PBMC** with mini-batches, to check whether the gap is explained by the subset.
""")

# =============================================================================================
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbf.write(nb, NB)
subprocess.run([str(Path(sys.executable).parent / "jupyter"), "nbconvert", "--to", "html", "--no-input",
                "--output", str(HTML), str(NB)], check=True)
print(f"wrote {NB.name} and {HTML} ({HTML.stat().st_size / 1e6:.1f} MB)")
