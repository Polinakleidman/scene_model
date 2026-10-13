# Joint cell–gene–protein embeddings for interpreting drug responses: a stress test of SCENE

- **Paper:** Møberg et al., *Setting the SCENE for Interpretable Cell–Gene Embeddings in Single-Cell RNA-seq*, bioRxiv (2026) — https://www.biorxiv.org/content/10.64898/2026.09.12.750699v1
- **Code:** https://github.com/oscarmoeberg/SCENE (`pip install scene-ldm`) · reproducibility repo: https://github.com/oscarmoeberg/SCENE-reproducibility
- **Data:** all public (GEO / scvi-tools / pertpy loaders), no access requests needed.

## What SCENE does, intuitively

SCENE places every **cell and every gene as points in the same small space** (even 2D or 3D). A gene is likely to be expressed in a cell, and at a higher level, if the two points are close. This works from raw counts and uses all genes, with no normalization and no "top-2000 variable genes" filtering. So you can read biology straight off the map: marker genes sit next to their cell types, and genes that respond together should cluster together.

## Our project in one line

We extend SCENE from one modality to two (**RNA + surface proteins**) and use it to ask whether **drug responses and drug targets** form readable structure in the embedding. We also check honestly how much of that structure is real biology and how much is technical artifact.

## Fit with the course (CBIO7215, Multimodal AI for therapeutic discovery)

| Course requirement | How we address it |
|---|---|
| **Multimodal** | RNA + protein (CITE-seq) in one shared embedding with modality-specific likelihoods. The SCENE authors list this as untested future work. |
| **Therapeutic-discovery problem** | Drug-response interpretation: do the genes that respond to a drug, and its known targets (Open Targets / DrugBank), move together in the embedding? Can the embedding recognize a drug's mechanism of action (MoA)? |
| **Rigor over performance; negative results OK** | Pre-registered hypotheses, a fair expression-matched null test, an audit for technical artifacts, several seeds, and baselines. |
| **Public / approved data** | Datasets listed below. We will confirm them with the instructor before starting. |

## Datasets (all public)

| Dataset | Modality | Why |
|---|---|---|
| PBMC CITE-seq (Hao et al. 2021) | RNA + 224 proteins | Main multimodal dataset; 8 donors → leave-one-donor-out tests |
| NeurIPS 2021 bone marrow CITE-seq | RNA + 134 proteins | Second multimodal dataset; differentiation lineages |
| IFN-β stimulation (Kang et al. 2018) | RNA | Interferon-β is an approved drug; response program is well characterized |
| Glucocorticoid time course (dexamethasone) | RNA | Drug response over time (1–18 h), used in the SCENE paper |
| sci-Plex (Srivatsan et al. 2020), subset | RNA | Cancer cell lines treated with 188 compounds with annotated MoA |

## Timeline

- **Saturday:** set everything up, write the code, smoke-test it, and launch the training queues on both laptops in the evening.
- **Weekdays (~2 days of compute):** runs go unattended. Check the log once a day.
- **Next weekend:** analyse the results, combine the two tracks, and write up.

## Who does what

| | **Person A: Multimodal model (ML track)** | **Person B: Drug response and validation (biology + stats track)** |
|---|---|---|
| **Big idea** | Make SCENE **multimodal**. Add proteins as a third type of point, with their own count model (NB). Also test better count models for RNA and a way to embed new patients (donors) without retraining. | Test whether the embedding is **useful and trustworthy for therapeutic questions**. Check whether drug-response genes and drug targets form coherent groups, and whether that holds up against a fair null test and technical confounders. |
| **Sat, together (~1.5 h)** | Shared repo and environment; download datasets; time one training run; write down hypotheses *before* running. | Same session. |
| **Sat, task 1** | **Protein modality (~2.5 h):** add ADT counts as extra nodes with an NB likelihood and a per-protein background term. Cells, genes and proteins then share one space. | **Baseline embeddings (~1.5 h):** train default RNA-only SCENE on IFN-β, dexamethasone and a sci-Plex subset (3 seeds). Compute baselines: PCA loadings, LDVAE (linear scVI), GenePT. |
| **Sat, task 2** | **RNA likelihood variants (~1.5 h):** Negative Binomial and a "decoupled" variant (separate parameters for "detected?" vs. "how much?"). | **Drug gene sets (~1.5 h):** IFN-β and GR response genes, targets of the sci-Plex compounds from Open Targets / DrugBank, MoA classes. Technical control sets: ribosomal, mitochondrial, cell cycle, dissociation stress, high-dropout genes. |
| **Sat, task 3** | **New donors (~1.5 h):** "fold-in" (freeze genes and proteins, fit only the new cells) vs. a zero-training baseline: cell = expression-weighted average of its gene positions. | **Fair null test (~2 h):** the paper compares modules against *uniformly random* genes. We compare against random genes *matched on expression level and detection rate*. |
| **Sat, task 4** | **Experiment configs (~1 h):** {RNA-only, RNA+protein} × {ZIP, NB, decoupled} × 2 dims (3D, 16D) × 2 CITE-seq datasets × 3 seeds, plus leave-one-donor-out on PBMC vs. totalVI. | **Artifact audit (~1.5 h):** how much of a gene's position is explained by technical covariates (mean expression, dropout, variance)? Measured with R² and kNN purity. |
| **Sat evening, together (~1 h)** | Smoke test: 5 epochs on a small subset → launch the queue (`caffeinate`, laptop plugged in). | Same; launch own queue. |
| **Weekdays** | Check the log once a day (~5 min). | Same. Tasks 3–4 can also be finished here; they are light computations on top of the saved embeddings. |
| **Next Sat (~4–5 h)** | Tables and plots. Does protein help separate similar cell types? Does each protein land next to its own gene (CD8 protein ↔ *CD8A*)? Held-out likelihood, integration metrics, label-transfer accuracy for new donors. | **MoA test:** compute each drug's "shift" in gene space (treated vs. control) and check whether drugs with the same MoA shift similarly (sci-Plex). Run the null test and artifact audit on all embeddings, including Person A's variants. |
| **Next Sun, together (~4 h)** | **Joint question:** does adding protein, or a better likelihood, make drug-response structure clearer and less technical? Write Methods and Results (A). | Write Intro (therapeutic motivation), Results (B) and Discussion (limitations). |
| **Total** | ~10–11 h active work | ~10–11 h active work |

## How the two tracks connect

- **Interface = `.h5ad` files.** Every trained model saves cell positions in `adata.obsm`, gene and protein positions in `adata.varm`, plus a `metrics.json`. Person B's analysis code runs on any such file, so it works on Person A's models without extra effort.
- **No waiting on each other:** each person trains on their own laptop.
- **Possible headline result:** *"A multimodal joint embedding makes drug-response programs readable and robust to technical artifacts"*. The opposite finding would also be worth reporting.

## Deliverables

A 10-minute presentation (7 min talk + 3 min Q&A), a technical report, and a documented, reproducible code repository.

## Main risks

- **Training time** on a laptop is unknown until we measure it on Saturday morning. If one run takes more than ~40 min, we shrink the grid (fewer cells, epochs or seeds).
- **sci-Plex is large.** We use a subset of cell lines and doses.
