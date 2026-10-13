# `scene_mm`: multimodal SCENE (ML track, Person A)

An extension of SCENE ([Møberg et al. 2026](https://www.biorxiv.org/content/10.64898/2026.09.12.750699v1)) that
(1) adds **surface proteins (CITE-seq ADT)** as a third type of point, (2) tests **other count likelihoods** for
RNA, and (3) embeds **new donors without retraining**. The authors' code is not modified.

## Model

Every cell has one position $z_i$ shared by all modalities. Each modality has its own feature positions
(genes $w_j$, proteins $v_k$), cell and feature effects, and batch effects:

$$\eta_{ij} = \psi_i + \rho_j + \sum_\ell g^{(\ell)}_{b_\ell(i),j} - \lVert z_i - w_j\rVert \qquad \text{(Euclidean, not squared, as in the authors' code)}$$

The paper writes the gene effect as $\omega_j$; we write $\rho_j$ so that it is not confused with the gene position $w_j$.

| Modality | Likelihood | Parameters |
|---|---|---|
| RNA `zip` | authors' hurdle ZIP: $P(x>0)=\sigma(\alpha\eta)$, positives $\sim$ zero-truncated Poisson($e^\eta$) | as in SCENE |
| RNA `nb` | negative binomial, mean $e^\eta$, one dispersion $\theta_j$ per gene | + $\theta_j$ |
| RNA `zip_decoupled` | as `zip`, but detection uses its own effects: $P(x>0)=\sigma(\alpha(\psi^\pi_i+\rho^\pi_j+g-d_{ij}))$ | + $\psi^\pi, \rho^\pi$ |
| RNA `poisson` | plain Poisson (sanity check) | — |
| Protein `nb_bg` | NB, mean $\mu_{ik}=e^{\psi_i}\,(e^{\beta_k} + e^{\rho_k - \lVert z_i - v_k\rVert})$: background + foreground | + $\beta_k$, $\theta_k$ |

The protein background $\beta_k$ models ambient antibody. A protein far from a cell stays at its background
level, which is why proteins need their own likelihood (only ~8–10 % of ADT entries are zero, mean count ~25–50).

**Training** follows the authors: AdamW, lr 0.05, weight decay 1e-3 on positions and batch terms, 600 epochs,
full-batch gradients (computed in blocks of cells and accumulated, so memory stays bounded). The expensive terms
(`lgamma`, `log expm1`) are evaluated only at nonzero counts.

**Check against the authors' implementation.** RNA-only `zip` on the same 2 000 PBMC cells (16D, 600 epochs, 2 seeds):
AUC / PR-AUC / F1 agree to 3–4 decimals (0.9235 vs 0.9236), and Poisson deviance falls within the seed-to-seed
range (0.54–0.58 for both).

**New cells.** `aggregate()` is a zero-training baseline: a cell is the mean of its features' positions,
weighted by count divided by the reference mean of that feature. `fold_in()` freezes everything on the feature side
(positions, effects, dispersions, background, alpha, known batch rows) and optimises only the new cells, plus a
new batch row for the unseen donor.

## Files

| Path | What |
|---|---|
| `scene_mm/model.py` | `MultiSCENE`, likelihoods, batch effects |
| `scene_mm/data.py` | counts per modality, held-out masks (RNA: authors' protocol; protein: 10 % of all entries treated as missing) |
| `scene_mm/train.py` | `fit()`, held-out evaluation, `export()` to `.h5ad` |
| `scene_mm/foldin.py` | `aggregate()`, `fold_in()` |
| `scene_mm/evaluate.py` | kNN label transfer, KMeans NMI/ARI, silhouette, protein ↔ gene alignment (ADT → gene map) |
| `scripts/prepare_data.py` | builds `data/pbmc_mm_sub30k.h5ad` (the 30k reproduction subset, 217 ADT) and `data/neurips_mm_sub30k.h5ad` (30k random cells, 134 ADT) |
| `scripts/run_experiment.py` | one run: `--mode full` or `--mode lodo` |
| `scripts/run_baseline.py` | scVI / totalVI, full or LODO via scArches |
| `scripts/make_grid.py`, `scripts/run_grid.sbatch` | the experiment grid, 2 workstation GPUs |
| `tests/test_scene_mm.py` | unit tests (`python -m pytest -q tests`) |

Outputs are in `results/<run_id>/`: `metrics.json`, `embedding.h5ad` (cells in `obsm["X_scene"]`, genes in
`varm["W_rna"]`, proteins in `uns["protein_positions"]`), `model.pt`, and `query_embedding.h5ad` for LODO.
This is the `.h5ad` interface that Person B's analysis reads.

## Experiment grid (132 runs, ~12 h on 2 × RTX 5000 Ada)

| Block | Runs |
|---|---|
| full | {RNA, RNA+protein} × {zip, nb, zip_decoupled} × {3D, 16D} × {PBMC, NeurIPS} × 3 seeds = 72 |
| LODO (PBMC) | 8 donors × {RNA, RNA+protein} × {zip, nb}, 16D = 32 |
| baselines | scVI, totalVI: full (2 datasets × 3 seeds) + PBMC LODO (8 donors) = 28 |

## Pre-registered hypotheses (written 2026-09-28, before any grid result)

- **H1 (likelihood).** NB gives a better held-out log-likelihood for RNA than ZIP, because it models overdispersion
  explicitly. Link-prediction AUC will be similar. The decoupled ZIP will improve detection (AUC) slightly over ZIP.
- **H2 (proteins help cell types).** Adding proteins improves clustering of fine cell types (PBMC `celltype.l2`,
  NeurIPS `cell_type`; NMI/ARI), and above all separates the T/NK subtypes that surface markers define.
- **H3 (protein ↔ gene).** In the joint space, each protein lies closer to its own gene than a random gene does:
  the median percentile rank of the own gene is well below 0.5.
- **H4 (new donors).** Fold-in transfers labels to an unseen donor almost as well as the reference itself, and better
  than aggregation. Aggregation already beats chance by a wide margin (seen in the smoke test: 0.76 vs 0.93 on 6k cells).
- **H5 (vs. totalVI).** On label transfer, totalVI is a strong baseline. We expect multimodal SCENE to be comparable,
  not clearly better. Its advantage should be interpretability (genes and proteins in the same space), not accuracy.

A result that contradicts a hypothesis is reported as such.

## Results (full analysis: `trackA_report.html`)

| | Outcome |
|---|---|
| H1 | ✅ NB has the best held-out log-likelihood; AUC is the same for all. ❌ `zip_decoupled` does not help. NB's worse Poisson deviance comes from counts ≥ 20: NB puts large counts in its heavy tail instead of raising the mean, and Poisson deviance assumes variance = mean. |
| H2 | ✅ moderate: proteins raise fine cell-type NMI by +0.01–0.06 (PBMC 0.645→0.679 at best) |
| H3 | ✅ strong: median percentile of the own gene is 0.4–3 % (16D). Failures are proteins without signal over background (Spearman ≈ −0.65 with the foreground share) |
| H4 | ✅ fold-in ≫ aggregation (0.90 vs 0.48, PBMC `celltype.l2`); proteins improve fold-in 0.83 → 0.90 |
| H5 | ✅ totalVI 0.914 / 0.785 (acc / macro-F1) vs multimodal SCENE 0.896 / 0.726; both beat scVI (0.879 / 0.712) |
