#!/bin/bash
# Toy run of multimodal SCENE on the shipped 2,000-cell PBMC CITE-seq subset (toy_data/), CPU, no downloads.
#   1. RNA only vs RNA + protein (NB likelihood, protein weight 3): held-out NLL, cell-type NMI, protein<->gene ranks
#   2. leave one donor out: fold-in vs aggregation for the held-out donor
# Results go to results_toy/ (~1 min on a laptop CPU). Effects are weaker than on the 30k-cell data but go the same way.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
EP=${EPOCHS:-600}
OUT=results_toy
COMMON="--dataset toy --rna-lik nb --dim 16 --epochs $EP --device cpu --out $OUT"

$PY scripts/run_experiment.py $COMMON --mode full
$PY scripts/run_experiment.py $COMMON --mode full --protein --protein-weight 3
$PY scripts/run_experiment.py $COMMON --mode lodo --lodo-donor P1 --protein --protein-weight 3 --foldin-epochs 200

$PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
for run in sorted(out.glob("toy_full_*")):
    m = json.loads((run / "metrics.json").read_text())
    line = f"{run.name:36s} held-out NLL {m['heldout']['rna']['heldout_nll']:.3f}   NMI (fine types) {m['clustering']['celltype.l2']['nmi']:.3f}"
    if "protein_gene" in m:
        line += f"   protein->own gene median percentile {m['protein_gene']['median_rank_pct']:.3f}"
    print(line)
for run in sorted(out.glob("toy_lodo_*")):
    t = json.loads((run / "metrics.json").read_text())["transfer"]["celltype.l2"]
    print(f"{run.name:36s} new-donor label transfer: aggregation {t['aggregate']['accuracy']:.3f}, fold-in {t['foldin']['accuracy']:.3f}")
EOF
