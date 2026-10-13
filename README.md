# Multimodal SCENE: reproduction and extension to CITE-seq

Course project for CBIO7215 *Multimodal AI for Therapeutic Discovery* (MBZUAI, 2026).

This repository contains two things:

1. **A reproduction** of [SCENE](https://www.biorxiv.org/content/10.64898/2026.09.12.750699v1)
   (Møberg et al., bioRxiv 2026), a latent-distance model that puts cells and genes on one interpretable map.
2. **`scene_mm`**, our extension of SCENE to **RNA + surface proteins (CITE-seq)**, with new count
   likelihoods and a way to add new donors without retraining.

## The model in one paragraph

Each cell $i$, gene $j$ and protein $k$ is a point ($z_i$, $w_j$, $v_k$) in one shared space.
RNA and protein counts are negative binomial with means

$$
\mu_{ij} = \exp\big(\psi_i + \rho_j + g_{b(i),j} - \lVert z_i - w_j \rVert\big),\qquad
\mu_{ik} = e^{\psi'_i + g_{b(i),k}}\Big(e^{\beta_k} + e^{\rho_k - \lVert z_i - v_k \rVert}\Big),
$$

where $\psi$ is cell depth, $\rho$ the gene/protein level, $g$ a (low-rank) batch offset and $e^{\beta_k}$
the antibody background. The loss is $-\sum \log P(\text{RNA}) - w\sum\log P(\text{protein})$ ($w=3$ works best),
optimised with full-batch AdamW. The original SCENE is the RNA part with a zero-inflated Poisson (hurdle) likelihood.
Full derivations: [docs/SCENE_math.md](docs/SCENE_math.md).

## Results

### Reproduction of the paper (9 of 10 claims hold)

| Claim | Paper | Our rerun |
|---|---|---|
| Nested batches (SIM2), NMI | SCENE ≫ scVI | 0.99 vs 0.29 |
| PBMC clusters match cell types, NMI | 0.63 | 0.66 (30k-cell subset) |
| GR (dexamethasone) modules, q-value | SCENE < SIMBA | 1e-4 to 1e-3 vs ~0.01 |
| T-cell activation regulons | SCENE > SIMBA | 58 vs 42 significant |
| Imputation, Pearson r | SCENE > scVI | 0.85 vs 0.65 |
| Better clustering on real atlases | competitive or superior | on par; SIMBA/Harmony slightly ahead |

### Multimodal extension (pre-registered hypotheses, PBMC + NeurIPS 2021 BMMC CITE-seq, 160 runs)

| Hypothesis | Result |
|---|---|
| H1: NB fits RNA better than ZIP | Supported: held-out NLL 1.173 vs 1.190 (PBMC), 1.226 vs 1.307 (NeurIPS). A "decoupled" ZIP did not help |
| H2: proteins sharpen fine cell types | Supported, moderate: NMI 0.62 → 0.70 (protein weight 3; totalVI 0.71) |
| H3: each protein lands next to its own gene | Supported, strong: median own-gene rank in the top 0.4–3% of ~20k genes; isotype controls are 99% background |
| H4: fold-in beats no-training placement for new donors | Supported: label transfer 0.90 vs ≈ 0.5 |
| H5: comparable to totalVI, not better | Supported: 0.918 vs 0.916 (leave-one-donor-out, weight 3) |
| Extra: predict proteins of a new donor from RNA | Negative: SCENE 0.43–0.47 vs kNN 0.49 vs totalVI 0.53 (Pearson r) |

Details and figures: [trackA_report.html](trackA_report.html), [scene_mm/README.md](scene_mm/README.md),
[docs/STATUS.md](docs/STATUS.md). The full reproduction report (31 MB, 113 figures) is attached to the GitHub release.

## Repository layout

```
scene_mm/                    multimodal SCENE: model, data, training, fold-in, evaluation
scripts/                     data preparation, experiment runner, baselines, analyses, report builders
experiments/                 run lists used on the cluster (one line per run)
tests/                       unit tests for scene_mm (pytest)
results_tables/              collected metrics (CSV) behind every number above
results_protein/             protein-prediction results per donor and method
figures_en/                  figures used in the slides
reproduction/                scripts to rerun the authors' pipeline (Mac and SLURM)
  SCENE-reproducibility/     authors' code (git submodule, pinned commit)
  upstream_changes.patch     our small fixes to the authors' code
docs/                        project plan, status, math derivations
```

## How to reproduce

```bash
git clone --recursive https://github.com/Polinakleidman/scene_model
cd scene_model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Reproduction of the paper (downloads data, runs the authors' scripts)
git -C reproduction/SCENE-reproducibility apply ../upstream_changes.patch
# see reproduction/README.md for Mac and SLURM instructions

# 2. Multimodal extension
python scripts/prepare_data.py all            # uses the raw files downloaded in step 1
python scripts/make_grid.py cuda               # writes experiments/grid_worker*.txt
sbatch --export=ALL,GRID=grid,WORKER=0 scripts/run_grid.sbatch   # and WORKER=1
python scripts/collect_results.py              # -> results_tables/
python scripts/analyze_likelihood.py mps
python scripts/analyze_proteins.py mps
python scripts/build_trackA_report.py          # -> trackA_report.html
pytest tests/
```

A single run: `python scripts/run_experiment.py --help`.
Every trained run writes `results/<run>/embedding.h5ad` with cell positions in `obsm["X_scene"]`,
gene positions in `varm["W_rna"]` and protein positions in `uns["protein_positions"]`.

## Credits

- SCENE model and reference code: Oscar Lauritz Møberg et al., MIT License
  ([oscarmoeberg/SCENE-reproducibility](https://github.com/oscarmoeberg/SCENE-reproducibility)).
- Baselines: scVI and totalVI from [scvi-tools](https://scvi-tools.org).
- Data: PBMC CITE-seq (Hao et al. 2021); NeurIPS 2021 multimodal integration challenge, BMMC CITE-seq (GSE194122).
- The drug-response analysis of the project (Track B) is not part of this repository.

License: MIT (see [LICENSE](LICENSE)).
