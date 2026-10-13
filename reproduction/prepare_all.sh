#!/usr/bin/env bash
# Download + prepare all paper datasets except TCR (needs R).
# Large files are fetched with curl (retries + resume) because the authors'
# urllib downloader has no timeout and can hang. Safe to re-run.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/SCENE-reproducibility"
PY="$ROOT/.venv/bin/python"
mkdir -p data/raw "$ROOT/logs"

fetch() {  # url dest
  [ -f "$2" ] && return 0
  echo "curl $1"
  curl -L --fail --retry 10 --retry-delay 5 --speed-limit 1000 --speed-time 60 -C - -o "$2.part" "$1" && mv "$2.part" "$2"
}

rm -f data/raw/*.part
fetch https://ndownloader.figshare.com/files/33798263 data/raw/sim1_1_norm.h5ad
fetch https://ndownloader.figshare.com/files/33798764 data/raw/sim2_norm.h5ad
NEURIPS=GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad
if [ ! -f "data/raw/$NEURIPS" ]; then
  fetch "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/$NEURIPS.gz" "data/raw/$NEURIPS.gz" \
    && gunzip -f "data/raw/$NEURIPS.gz"
fi

for d in cortex gr kang sim1 sim2 neurips hca pbmc; do
  case $d in
    cortex) out=data/cortex.h5ad ;; gr) out=data/GR_18.h5ad ;; kang) out=data/kang_2018.h5ad ;;
    sim1) out=data/sim1_1_norm.h5ad ;; sim2) out=data/sim2_norm.h5ad ;;
    neurips) out=data/neurips_cite_gex_stem_cells.h5ad ;; hca) out=data/hca_nuclei.h5ad ;;
    pbmc) out=data/pbmc_cite_seq.h5ad ;;
  esac
  [ -f "$out" ] && { echo "skip $d ($out exists)"; continue; }
  echo "=== $d $(date +%T)"
  if [ "$d" = pbmc ]; then  # authors' version fails on '/' in a protein name with current anndata
    "$PY" "$ROOT/prepare_pbmc_fix.py" 2>&1 | grep -v -i warning | tail -4; continue
  fi
  "$PY" scripts/prepare_data.py "$d" 2>&1 | grep -v -i warning | tail -4
done
echo "DONE $(date +%T)"
ls -la data
