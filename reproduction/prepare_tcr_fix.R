# Replacement for SCENE-reproducibility/scripts/prepare_tcr.R.
# With current SeuratObject, Seurat::as.SingleCellExperiment() fails (PackageCheck() is defunct),
# so we export the RNA counts and cell metadata directly; prepare_tcr_fix.py builds the same .h5ad.
suppressPackageStartupMessages({ library(Seurat); library(Matrix) })
args <- commandArgs(trailingOnly = TRUE)
raw_dir <- if (length(args)) args[1] else "data/raw"
for (sample in c("GSM4450386_stimulated", "GSM4450387_unstimulated")) {
    obj <- UpdateSeuratObject(readRDS(file.path(raw_dir, paste0(sample, "_full_seurat.rds"))))
    counts <- LayerData(obj, assay = "RNA", layer = "counts")
    prefix <- file.path(raw_dir, sample)
    writeMM(counts, paste0(prefix, "_counts.mtx"))
    writeLines(rownames(counts), paste0(prefix, "_genes.txt"))
    write.csv(obj@meta.data, paste0(prefix, "_obs.csv"))
    cat(sample, ":", nrow(counts), "genes x", ncol(counts), "cells\n")
}
