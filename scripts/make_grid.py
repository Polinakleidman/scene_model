"""Write the experiment grid of the ML track as two balanced command lists (one per GPU worker):
experiments/grid_worker0.txt, experiments/grid_worker1.txt.

  full      : {RNA, RNA+protein} x {zip, nb, zip_decoupled} x {3D, 16D} x {pbmc, neurips} x 3 seeds
  lodo      : PBMC, 8 donors x {RNA, RNA+protein} x {zip, nb}, 16D, seed 0
  baselines : scVI, totalVI -- full (2 datasets x 3 seeds, 16D) and PBMC LODO (8 donors)
Every script skips runs whose metrics.json already exists, so the lists can be re-run safely.
"""
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
ACCEL = "gpu" if DEVICE == "cuda" else DEVICE
PY = "python"
DONORS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
# rough minutes on one RTX 5000 Ada (30k cells, 600 epochs), used only to balance the two workers
COST = {"pbmc": 14, "neurips": 9}

jobs = []
for ds in ["pbmc", "neurips"]:
    for prot in [False, True]:
        for lik in ["zip", "nb", "zip_decoupled"]:
            for dim in [3, 16]:
                for seed in [0, 1, 2]:
                    cmd = (f"{PY} scripts/run_experiment.py --dataset {ds} --mode full --rna-lik {lik} "
                           f"--dim {dim} --seed {seed} --device {DEVICE}" + (" --protein" if prot else ""))
                    jobs.append((COST[ds] * (1.3 if lik == "nb" else 1.0), cmd))
for donor in DONORS:
    for prot in [False, True]:
        for lik in ["zip", "nb"]:
            cmd = (f"{PY} scripts/run_experiment.py --dataset pbmc --mode lodo --lodo-donor {donor} "
                   f"--rna-lik {lik} --dim 16 --seed 0 --device {DEVICE}" + (" --protein" if prot else ""))
            jobs.append((COST["pbmc"] * 0.9 + 1, cmd))
for method in ["scvi", "totalvi"]:
    for ds in ["pbmc", "neurips"]:
        for seed in [0, 1, 2]:
            jobs.append((10, f"{PY} scripts/run_baseline.py --dataset {ds} --method {method} --mode full "
                             f"--dim 16 --seed {seed} --accelerator {ACCEL}"))
    for donor in DONORS:
        jobs.append((10, f"{PY} scripts/run_baseline.py --dataset pbmc --method {method} --mode lodo "
                         f"--lodo-donor {donor} --dim 16 --seed 0 --accelerator {ACCEL}"))

# greedy balancing: longest job first onto the less loaded worker
workers = [[], []]
load = [0.0, 0.0]
for cost, cmd in sorted(jobs, key=lambda j: -j[0]):
    w = load.index(min(load))
    workers[w].append(cmd)
    load[w] += cost
(PROJ / "experiments").mkdir(exist_ok=True)
for i, cmds in enumerate(workers):
    (PROJ / "experiments" / f"grid_worker{i}.txt").write_text("\n".join(cmds) + "\n")
print(f"{len(jobs)} runs; estimated hours per worker: {[round(l / 60, 1) for l in load]}")
