# Reproducing the SCENE paper on a MacBook (M5 Pro, 48 GB)

This folder reruns (almost) every experiment from
[Møberg et al. 2026](https://www.biorxiv.org/content/10.64898/2026.09.12.750699v1),
using the authors' own scripts from
[SCENE-reproducibility](https://github.com/oscarmoeberg/SCENE-reproducibility) (cloned in `SCENE-reproducibility/`).

## Layout

| Path | What |
|---|---|
| `SCENE-reproducibility/` | Authors' repo, unmodified. Data goes to `data/`, outputs to `results/`, figures to `notebooks/figures/output/` |
| `.venv/` | Python 3.13 env (torch with MPS, scanpy, scvi-tools, harmonypy, …) plus their `scene` package |
| `prepare_all.sh` | Downloads and prepares all datasets except TCR (curl with retries, because their downloader can hang) |
| `run_queue.py` | Runs every command from `bash/paper/*.sh` as a resumable queue, with Mac adaptations |
| `logs/queue.log` | One line per command: START / OK / FAIL, with timing. **Check this file** |
| `logs/commands.log` | Full stdout/stderr of every command |
| `logs/done.txt`, `logs/failed.txt` | Finished commands (skipped on re-run) and failed ones |

## Toy run (~1 minute, no downloads)

`./toy_run.sh` trains SCENE (16D, ZIP, 200 epochs) and the PCA baseline with the authors' scripts on their bundled
cortex data (3,005 cells × 19,972 genes) and prints KMeans NMI/ARI. Expected: SCENE NMI 0.84, PCA 0.69.

## How to run

```bash
./prepare_all.sh                                # ~1–2 h, mostly downloads
caffeinate -i .venv/bin/python run_queue.py     # resumable; re-run anytime to continue
.venv/bin/python run_queue.py --dry-run         # show what is left
PBMC_MODE=full caffeinate -i .venv/bin/python run_queue.py   # optional: full PBMC (~+35 h)
```

Keep the laptop plugged in. `caffeinate -i` stops it from sleeping while the queue runs.

## Running on a Linux GPU cluster (SLURM)

The archive `scene_reproduction.tar.gz` contains the code and the **prepared** datasets (`data/*.h5ad`), so nothing needs to be downloaded again. It does not include the raw downloads, the Mac `.venv` or any results.
If you already have an older copy, unpack `scene_reproduction_patch.tar.gz` (small, scripts only) over it: `tar xzf scene_reproduction_patch.tar.gz` in the folder that contains `reproduction/`.

```bash
tar xzf scene_reproduction.tar.gz && cd reproduction
./setup_cluster.sh                                   # on the login node (pip needs internet)
sbatch --export=ALL,PART=scvi run_cluster.sbatch     # only scVI (the Mac runs everything else)
# or: sbatch run_cluster.sbatch                      # everything, full PBMC
squeue -u $USER;  tail -f logs/queue.log             # status / progress;  scancel -n scene_repro  to stop
```

- Each job is 1 GPU, 8 CPUs, 64 GB, 8 h. If the work takes longer, each job queues its own continuation (`--dependency=afterany`) and the queue resumes where it stopped. A command cut off by the time limit is simply rerun.
- **Split mode (`PART=scvi`)** runs the 33 scVI trainings and 30 scVI evaluations on the same 30k-cell PBMC subset as the Mac (rebuilt from `pbmc_sub30k_cells.txt`). The Mac runs the rest with `run_queue.py --exclude` using the same regex. When done, copy the results back to the Mac:
  `rsync -a mbzu-login:~/reproduction/SCENE-reproducibility/results/ SCENE-reproducibility/results/`
- **Full mode** uses all 152k PBMC cells; SCENE trains with mini-batches of `CELL_BATCH=8192` cells. Lower it if the GPU has much less than 24 GB.

## Differences from the paper (and why)

| Change | Reason |
|---|---|
| SCENE on Apple GPU (`--device mps`) instead of CPU/CUDA | ~2× faster than CPU here; small numerical differences are expected |
| scVI also on MPS | ~3× faster than CPU (2.5 vs 7.5 s/epoch on HCA) |
| **SIMBA skipped** | Needs a separate environment with old dependencies |
| **TCR dataset skipped** | Data preparation needs R/Seurat |
| **PBMC uses a fixed 30k-cell subset** (seed 0) | Dense training needs ~41 bytes per cell×gene entry (measured), so full PBMC would need ~130 GB of RAM |
| Full NeurIPS (90k cells): SCENE with cell mini-batches (8192) | Dense training would need ~52 GB |
| PBMC protein name `CD66a/c/e` → `CD66a_c_e` (`prepare_pbmc_fix.py`) | Current anndata cannot write '/' in keys; RNA counts are unchanged |
| **Dimension sweep (`dimensions.sh`, Fig. 5e) with seeds 0–1 instead of 0–4**; `fig_5_ultra_low_dim.ipynb` set to `seeds = [0, 1]` | Time budget. The figure still shows mean ± spread, over 2 seeds |
| **Where each part ran.** Mac (MPS, full-matrix SCENE): 74 commands. Cluster (RTX 5000 Ada): all scVI, all PBMC, and everything left after 07:30 on 26 Sep | The laptop was needed for other work |
| **Cluster SCENE fits use mini-batches of 2048 cells** (`PART=rest`: `MINIBATCH_ALL=1`) | The GPU node is shared: other users' processes hold 26–32 GB on most GPUs, so full-matrix training runs out of memory |
| `scripts/imputation.py`: optional `SCENE_CELL_BATCH_SIZE` / `SCENE_RECON_CHUNK` env vars (3 lines) | Needed for PBMC imputation on the shared GPU. When unset, behaviour is identical to the authors' code |
| Finished work is moved between machines with `run_queue.py --export-done` / `--import-done` | `done.txt` keys depend on python path, device and flags |
| **SIMBA/TCR** (`WITH_SIMBA_TCR=1`, `setup_simba_tcr.sh`, `run_simba.sbatch`): `dgl==2.2.0` installed from data.dgl.ai; torch 2.3.0 CPU build; other versions as in `environment_simba.yaml` | dgl 2.2.0 is not on PyPI; SIMBA trains on CPU |
| TCR conversion via `prepare_tcr_fix.R` + `prepare_tcr_fix.py` instead of `scripts/prepare_tcr.R` | `Seurat::as.SingleCellExperiment` fails with current SeuratObject (`PackageCheck()` is defunct). The result matches the reference size, 9 492 × 17 840 |
| SCENE-TCR regulon evaluation run with `--n_jobs 12` (`finish_tcr_figs.sh`) | With `permutation_rng=independent`, each module has its own seed stream, so results do not depend on `n_jobs` (1 core: ~9 h) |
| `SCENE_full_report.html` (built by `build_full_report.py`, attached to the GitHub release) | Paper summary, architectures, metrics and all reproduced figures, with comments |

## Measured speed (MPS, 16D, ZIP)

| Dataset | Cells × genes | Time per epoch | 600 epochs | Peak memory |
|---|---|---|---|---|
| Cortex | 3k × 20k | 0.08 s | ~1 min | ~2 GB |
| Kang (IFN-β) | 25k × 16k | 0.48 s | ~5 min | ~16 GB |

**Expected total:** roughly 12–18 h for the full queue with PBMC as a subset. The PBMC commands run last.
