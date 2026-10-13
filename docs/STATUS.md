cltk

# Project status: multimodal SCENE

*Updated 2026-10-02. Full plan: [PROJECT_PLAN.md](PROJECT_PLAN.md). Track A details: [trackA_report.html](../trackA_report.html), [scene_mm/README.md](../scene_mm/README.md).*

## 0. Reproduction of the paper: done

We reran the authors' code: 9 of 10 claims hold.

- **Strongest claims:** nested batches (SIM2: NMI 0.99 vs 0.29 for scVI), GR/TCR gene biology, and imputation (Pearson 0.85 vs 0.65).
- **Where the claims are weaker:** on real datasets (PBMC, HCA, NeurIPS), SCENE clusters about as well as SIMBA/Harmony, not better.

Full report: SCENE_full_report.html (GitHub Releases).

## Track A (ML): done

**What we built:** `scene_mm`, an extension of SCENE where cells, genes **and surface proteins** (CITE-seq) are points in one shared 16-D space.

- RNA: choice of likelihood (ZIP, NB, decoupled, Poisson).
- Proteins: NB likelihood with a per-protein background term.
- New donors: fold-in (genes, proteins and batch effects frozen).
- Data: PBMC (217 proteins) and NeurIPS (134 proteins), 30k cells each.
- Runs: 160 runs on the cluster.

| Hypothesis (pre-registered)                 | Result                                                              |
| ------------------------------------------- | ------------------------------------------------------------------- |
| H1: NB fits RNA better than ZIP             | ✓ better held-out NLL. ✗ The decoupled variant gave nothing.      |
| H2: proteins help separate cell types       | ✓ PBMC NMI 0.62 → 0.70 (protein weight 3)                         |
| H3: each protein lands next to its own gene | ✓ median percentile 0.4–3%. Failures are proteins with no signal. |
| H4: fold-in beats a no-training baseline    | ✓ label transfer 0.90 vs 0.48                                      |
| H5: comparable to totalVI on new donors     | ✓ tie: 0.918 vs 0.916 (macro-F1 0.779 vs 0.777)                    |

**Extra checks**

- **Isotype controls:** the model explains about 99% of their counts as background. For real proteins, the signal share is a median of 42%.
- **Protein weight sweep:** weight 3 is best. It gives better clustering at almost the same RNA fit.
- **Predicting proteins of a new donor from RNA only (8 donors, Pearson):** totalVI 0.53 > kNN 0.49 > SCENE latent kNN 0.47 > SCENE NB 0.43 > SCENE ZIP 0.41.
  - **Negative result for SCENE.** Reading a protein from a single distance is too rigid. This is the price of interpretability.

**Small open items**

- Add an "own gene RNA" baseline for protein prediction.
- 3D works for plots, but fine structure needs 16D.

**Outputs for track B**

- Each run in `results/<run>/` has `embedding.h5ad`:
  - cells in `obsm`;
  - genes and proteins in `varm`;
  - `train_log.json` with metrics.
- Load a run with `scene_mm.train.load_run`.

## Track B (biology / statistics): not started

**Question:** do drug-response genes and drug targets form coherent groups in the embedding? Does this hold against a fair null and technical confounders?

1. **Embeddings.** Use existing embeddings: the reproduction already trained SCENE on GR/dexamethasone and TCR data, and track A provides PBMC/NeurIPS. Train new embeddings on IFN-β and a sci-Plex subset (3 seeds). Baselines: PCA loadings, LDVAE, GenePT.
2. **Gene sets.**
   - IFN-β and GR response genes.
   - Targets of sci-Plex compounds (Open Targets / DrugBank) and their MoA classes.
   - Technical control sets: ribosomal, mitochondrial, cell cycle, dissociation stress, high-dropout genes.
3. **Fair null test.** Compare module compactness against random genes **matched on expression level and detection rate**, not uniform random genes as in the paper.
4. **Artifact audit.** Measure how much of a gene's position is explained by mean expression, dropout and variance (R², kNN purity). Run it on every embedding, including the track A variants.
5. **MoA test (sci-Plex).** Compute each drug's shift in gene space (treated vs control). Check whether drugs with the same MoA shift similarly.

**Possible headline:** *"A multimodal joint embedding makes drug-response programs readable and robust to technical artifacts"*. The opposite finding is reportable too.

## Deliverables

- [X] Track A results and final presentation
- [ ] Track B results
- [ ] Final course report
