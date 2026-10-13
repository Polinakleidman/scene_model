# Toy data

`toy_pbmc_cite.h5ad` (5 MB) is a small CITE-seq dataset for running the code without downloads.

| | |
|---|---|
| Cells | 2,000: 250 random cells from each of the 8 donors of the PBMC CITE-seq data |
| Genes | 1,591: the 1,500 genes detected in most cells plus the 108 genes that code a measured protein |
| Proteins | 217 surface proteins (ADT counts) in `obsm["protein_counts"]` |
| Counts | raw integer counts in `X` |
| Labels | `obs["celltype.l1"]` (8 types), `obs["celltype.l2"]` (30 types), `obs["donor"]` (P1–P8) |

It was built by `scripts/make_toy_data.py` from the 30k-cell subset used in the project.
Run everything on it with `./toy_run.sh` (about 1 minute on a laptop CPU), or a single run with
`python scripts/run_experiment.py --dataset toy --protein --rna-lik nb --protein-weight 3 --device cpu`.

Expected output of `./toy_run.sh` (600 epochs):

| Run | Result |
|---|---|
| RNA only | held-out NLL 1.334, NMI (30 types) 0.630 |
| RNA + protein | held-out NLL 1.336, NMI 0.724, own-gene median percentile 0.34 (random: 0.5) |
| New donor P1 | label transfer: aggregation 0.276, fold-in 0.852 |

The effects go the same way as on the full data, but are weaker: with only 2,000 cells and 1,591 genes the
protein–gene match is far from the full-data value (median percentile 0.004–0.03).

## Raw data (full size)

| Dataset | Source | Used for |
|---|---|---|
| PBMC CITE-seq (Hao et al. 2021): 152,094 cells, 20,729 genes, 217 proteins | [GEO GSE164378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378); file used: [figshare 27458840](https://ndownloader.figshare.com/files/27458840) (scvi-tools `pbmc_seurat_v4_cite_seq`) | main multimodal dataset; we use a fixed 30k-cell subset (`reproduction/pbmc_sub30k_cells.txt`) |
| NeurIPS 2021 BMMC CITE-seq (Luecken et al. 2021): 90,261 cells, 13,953 genes, 134 proteins | [GEO GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122), file `GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz` | second multimodal dataset; 30k random cells |

`scripts/prepare_data.py` builds the two 30k-cell files from these sources.
The datasets of the original paper (cortex, HCA, Kang IFN-β, GR, TCR, SIM1, SIM2) are listed with links in the authors'
[docs/DATA.md](https://github.com/oscarmoeberg/SCENE-reproducibility/blob/main/docs/DATA.md).
