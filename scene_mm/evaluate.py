"""Evaluation: label transfer to new cells, clustering metrics, protein <-> gene alignment."""
from __future__ import annotations

import re

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, adjusted_rand_score, f1_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.neighbors import KNeighborsClassifier

# ADT (antibody) name -> gene symbol(s) for the TotalSeq panels in PBMC / NeurIPS CITE-seq.
# Only unambiguous, well-known pairs; proteins without a clear single gene (isotype controls, TCR-V
# segments, glycoforms) are left out. Clone suffixes like "-1"/"-2" are stripped before lookup.
PROTEIN_TO_GENE = {
    "CD2": ["CD2"], "CD3": ["CD3E", "CD3D", "CD3G"], "CD4": ["CD4"], "CD5": ["CD5"], "CD7": ["CD7"],
    "CD8": ["CD8A", "CD8B"], "CD8a": ["CD8A"], "CD9": ["CD9"], "CD11a": ["ITGAL"], "CD11b": ["ITGAM"],
    "CD11c": ["ITGAX"], "CD13": ["ANPEP"], "CD14": ["CD14"], "CD15": ["FUT4"], "CD16": ["FCGR3A"],
    "CD18": ["ITGB2"], "CD19": ["CD19"], "CD20": ["MS4A1"], "CD21": ["CR2"], "CD22": ["CD22"],
    "CD24": ["CD24"], "CD25": ["IL2RA"], "CD26": ["DPP4"], "CD27": ["CD27"], "CD28": ["CD28"],
    "CD29": ["ITGB1"], "CD31": ["PECAM1"], "CD32": ["FCGR2A", "FCGR2B"], "CD33": ["CD33"], "CD34": ["CD34"],
    "CD35": ["CR1"], "CD36": ["CD36"], "CD38": ["CD38"], "CD39": ["ENTPD1"], "CD40": ["CD40"],
    "CD41": ["ITGA2B"], "CD42b": ["GP1BA"], "CD44": ["CD44"], "CD45": ["PTPRC"], "CD45RA": ["PTPRC"],
    "CD45RB": ["PTPRC"], "CD45RO": ["PTPRC"], "CD47": ["CD47"], "CD48": ["CD48"], "CD49b": ["ITGA2"],
    "CD49d": ["ITGA4"], "CD49f": ["ITGA6"], "CD52": ["CD52"], "CD54": ["ICAM1"], "CD56": ["NCAM1"],
    "CD57": ["B3GAT1"], "CD58": ["CD58"], "CD61": ["ITGB3"], "CD62L": ["SELL"], "CD62P": ["SELP"],
    "CD64": ["FCGR1A"], "CD69": ["CD69"], "CD71": ["TFRC"], "CD73": ["NT5E"], "CD79b": ["CD79B"],
    "CD80": ["CD80"], "CD81": ["CD81"], "CD82": ["CD82"], "CD83": ["CD83"], "CD85j": ["LILRB1"],
    "CD86": ["CD86"], "CD88": ["C5AR1"], "CD94": ["KLRD1"], "CD95": ["FAS"], "CD96": ["CD96"],
    "CD99": ["CD99"], "CD101": ["CD101"], "CD103": ["ITGAE"], "CD105": ["ENG"], "CD107a": ["LAMP1"],
    "CD112": ["NECTIN2"], "CD115": ["CSF1R"], "CD117": ["KIT"], "CD122": ["IL2RB"], "CD123": ["IL3RA"],
    "CD124": ["IL4R"], "CD127": ["IL7R"], "CD134": ["TNFRSF4"], "CD137": ["TNFRSF9"], "CD141": ["THBD"],
    "CD146": ["MCAM"], "CD152": ["CTLA4"], "CD154": ["CD40LG"], "CD155": ["PVR"], "CD158": ["KIR2DL1"],
    "CD161": ["KLRB1"], "CD163": ["CD163"], "CD172a": ["SIRPA"], "CD178": ["FASLG"], "CD183": ["CXCR3"],
    "CD185": ["CXCR5"], "CD192": ["CCR2"], "CD194": ["CCR4"], "CD195": ["CCR5"], "CD196": ["CCR6"],
    "CD197": ["CCR7"], "CD223": ["LAG3"], "CD226": ["CD226"], "CD244": ["CD244"], "CD247": ["CD247"],
    "CD268": ["TNFRSF13C"], "CD270": ["TNFRSF14"], "CD273": ["PDCD1LG2"], "CD274": ["CD274"],
    "CD278": ["ICOS"], "CD279": ["PDCD1"], "CD294": ["PTGDR2"], "CD303": ["CLEC4C"], "CD304": ["NRP1"],
    "CD319": ["SLAMF7"], "CD335": ["NCR1"], "CD352": ["SLAMF6"], "CD357": ["TNFRSF18"], "CX3CR1": ["CX3CR1"],
    "HLA-DR": ["HLA-DRA"], "TIGIT": ["TIGIT"], "KLRG1": ["KLRG1"], "FceRIa": ["FCER1A"],
    "Galectin-9": ["LGALS9"], "IgD": ["IGHD"], "IgM": ["IGHM"], "TCR-2": ["TRDC"], "CD1c": ["CD1C"],
    "CD1d": ["CD1D"], "CLEC12A": ["CLEC12A"], "Siglec-8": ["SIGLEC8"], "CD340": ["ERBB2"],
}


def protein_gene_pairs(protein_names, gene_names):
    genes = {g: i for i, g in enumerate(gene_names)}
    pairs = []
    for pi, p in enumerate(protein_names):
        key = re.sub(r"-\d+$", "", str(p))
        for g in PROTEIN_TO_GENE.get(key, PROTEIN_TO_GENE.get(p, [])):
            if g in genes:
                pairs.append((pi, genes[g], str(p), g))
                break
    return pairs


def protein_gene_alignment(W_protein, W_rna, protein_names, gene_names):
    """For each protein with a known gene: percentile rank of its own gene among all genes by distance
    to the protein (0 = closest gene). Random placement gives a median of ~0.5."""
    pairs = protein_gene_pairs(protein_names, gene_names)
    ranks = []
    for pi, gi, _, _ in pairs:
        d = np.linalg.norm(W_rna - W_protein[pi], axis=1)
        ranks.append(float((d < d[gi]).mean()))
    ranks = np.asarray(ranks)
    return {"n_pairs": len(pairs), "median_rank_pct": float(np.median(ranks)) if len(ranks) else None,
            "frac_top1pct": float((ranks <= 0.01).mean()) if len(ranks) else None,
            "frac_top5pct": float((ranks <= 0.05).mean()) if len(ranks) else None,
            "per_pair": {f"{p}->{g}": r for (_, _, p, g), r in zip(pairs, ranks)}}


def label_transfer(Z_ref, y_ref, Z_q, y_q, k=15):
    knn = KNeighborsClassifier(n_neighbors=k).fit(Z_ref, y_ref)
    pred = knn.predict(Z_q)
    return {"accuracy": float(accuracy_score(y_q, pred)),
            "macro_f1": float(f1_score(y_q, pred, average="macro"))}


def cluster_metrics(Z, labels, seed=0, max_sil=5000):
    labels = np.asarray(labels)
    k = len(np.unique(labels))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(max_sil, len(Z)), replace=False)
    return {"nmi": float(normalized_mutual_info_score(labels, km)),
            "ari": float(adjusted_rand_score(labels, km)),
            "silhouette_label": float((silhouette_score(Z[idx], labels[idx]) + 1) / 2)}
