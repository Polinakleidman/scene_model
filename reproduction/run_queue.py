r"""Run the SCENE paper commands (bash/paper/*.sh) on a Mac, as a resumable queue.

Changes vs. the authors' scripts:
  * SCENE and scVI run on the Apple GPU (--device mps).
  * SIMBA and TCR commands are skipped (separate env / needs R).
  * PBMC (152k cells) is too large for dense full-matrix training in 48 GB RAM.
    PBMC_MODE=subset (default): every PBMC command uses a fixed 30k-cell subset.
    PBMC_MODE=full: full data, SCENE trained with cell mini-batches (~35 h extra).
  * Full NeurIPS (90k cells) also needs mini-batches (dense training would need ~52 GB).
  * Non-PBMC commands run first, PBMC commands last.

Finished commands are recorded in logs/done.txt, so re-running skips them.
Usage:  caffeinate -i .venv/bin/python run_queue.py [--dry-run] [--only SUBSTRING]
        DEVICE=cuda .venv/bin/python run_queue.py          # on a Linux GPU node
Split scVI to another machine (both sides must use the same regex):
        run_queue.py --include "$SCVI_RE"   /   run_queue.py --exclude "$SCVI_RE"
        SCVI_RE='train_scvi\.py|benchmark_embedding\.py .*--embedding_key scVI'

"""
import argparse
import datetime as dt
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "SCENE-reproducibility"
PY = ROOT / ".venv" / "bin" / "python"
LOGS = ROOT / "logs"
DONE = LOGS / "done.txt"
FAILED = LOGS / "failed.txt"

# WITH_SIMBA_TCR=1 adds the SIMBA scripts and the TCR commands (needs setup_simba_tcr.sh first);
# SIMBA training then runs with its own environment (envs/simba, python 3.10).
WITH_SIMBA_TCR = os.environ.get("WITH_SIMBA_TCR") == "1"
SIMBA_PY = ROOT / "envs" / "simba" / "bin" / "python"
SCRIPTS = ["figures.sh", "benchmarks.sh", "dimensions.sh", "imputation.sh",
           *(["figures_simba.sh", "benchmarks_simba.sh"] if WITH_SIMBA_TCR else []),
           "evaluate_benchmarks.sh", "evaluate_modules.sh", "figures_all.sh"]
SKIP = re.compile(r"(?!x)x" if WITH_SIMBA_TCR else r"simba|SIMBA|tcr_stim|TCR", re.I)

PBMC_MODE = os.environ.get("PBMC_MODE", "subset")
PBMC_SUBSET = int(os.environ.get("PBMC_SUBSET", "30000"))
PBMC_FULL = "data/pbmc_cite_seq.h5ad"
PBMC_SUB = f"data/pbmc_cite_seq_sub{PBMC_SUBSET // 1000}k.h5ad"
DEVICE = os.environ.get("DEVICE", "mps")  # "cuda" on a Linux GPU node
CELL_BATCH = os.environ.get("CELL_BATCH", "8192")
MINIBATCH_DATA = ["data/neurips_cite_gex.h5ad", PBMC_FULL]
# MINIBATCH_ALL=1: mini-batches for every SCENE fit (shared cluster GPUs have too little free memory).
MINIBATCH_ALL = os.environ.get("MINIBATCH_ALL") == "1"


def read_commands():
    cmds = []
    for name in SCRIPTS:
        for line in (REPO / "bash" / "paper" / name).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(("set ", "cd ")):
                continue
            cmds.append((name, line))
    return cmds


def transform(cmd):
    if cmd.startswith("python "):
        py = SIMBA_PY if "scripts/train_simba.py" in cmd else PY
        cmd = f"{py} " + cmd[len("python "):]
    # SCENE and scVI both run on the GPU (on Mac, scVI on MPS is ~3x faster than CPU, measured on HCA).
    cmd = re.sub(r"--device \S+", f"--device {DEVICE}", cmd)
    if "pbmc" in cmd and PBMC_MODE == "subset":
        cmd = cmd.replace(PBMC_FULL, PBMC_SUB)
    # Dense full-matrix training needs ~41 bytes per cell×gene entry (measured):
    # full NeurIPS (~52 GB) and full PBMC (~130 GB) exceed 48 GB, so use cell mini-batches.
    if "train_latent.py" in cmd and (MINIBATCH_ALL or any(d in cmd for d in MINIBATCH_DATA)):
        cmd += f" --cell_batch_size {CELL_BATCH}"
    return cmd


def make_pbmc_subset():
    out = REPO / PBMC_SUB
    if PBMC_MODE != "subset" or out.exists() or not (REPO / PBMC_FULL).exists():
        return
    import anndata as ad
    import numpy as np
    a = ad.read_h5ad(REPO / PBMC_FULL)
    # The cell list pins the exact subset used on the Mac, so other machines match it
    # regardless of their numpy RNG version.
    cells = ROOT / f"pbmc_sub{PBMC_SUBSET // 1000}k_cells.txt"
    if cells.exists():
        idx = np.sort(a.obs_names.get_indexer(cells.read_text().split()))
        assert (idx >= 0).all() and len(idx) == PBMC_SUBSET, "cell list does not match PBMC data"
        how = f"from {cells.name}"
    else:
        idx = np.sort(np.random.default_rng(0).choice(a.n_obs, PBMC_SUBSET, replace=False))
        how = "seed 0"
    a[idx].copy().write_h5ad(out)
    log(f"created {PBMC_SUB} ({PBMC_SUBSET} cells, {how})")


def key(cmd):
    return hashlib.sha1(cmd.encode()).hexdigest()[:16]


def log(msg):
    tag = f" [{os.environ['QUEUE_TAG']}]" if os.environ.get("QUEUE_TAG") else ""
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}]{tag} {msg}"
    print(line, flush=True)
    with open(LOGS / "queue.log", "a") as f:
        f.write(line + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", default=None, help="run only commands containing this substring")
    p.add_argument("--include", default=None, help="run only commands matching this regex")
    p.add_argument("--exclude", default=None, help="skip commands matching this regex")
    p.add_argument("--export-done", default=None, help="write finished commands (machine-independent form) to FILE")
    p.add_argument("--import-done", default=None, help="mark commands listed in FILE (from --export-done) as finished")
    args = p.parse_args()
    LOGS.mkdir(exist_ok=True)

    # done.txt keys depend on the machine (python path, device, flags); these two options
    # translate finished work between machines via the authors' original command lines.
    done = set(DONE.read_text().split()) if DONE.exists() else set()
    if args.export_done:
        orig = [c for _, c in read_commands() if key(transform(c)) in done]
        Path(args.export_done).write_text("\n".join(orig) + "\n")
        print(f"exported {len(orig)} finished commands to {args.export_done}")
        return
    if args.import_done:
        orig = [c for c in Path(args.import_done).read_text().splitlines() if c.strip()]
        new = [key(transform(c)) for c in orig if key(transform(c)) not in done]
        with open(DONE, "a") as f:
            f.writelines(k + "\n" for k in new)
        print(f"imported {len(orig)} finished commands ({len(new)} new)")
        return

    raw = [(s, c) for s, c in read_commands() if not SKIP.search(c)]
    if args.only:
        raw = [(s, c) for s, c in raw if args.only in c]
    if args.include:
        raw = [(s, c) for s, c in raw if re.search(args.include, c)]
    if args.exclude:
        raw = [(s, c) for s, c in raw if not re.search(args.exclude, c)]
    # Non-PBMC first, PBMC last; figure notebooks at the very end.
    order = lambda sc: (sc[0] == "figures_all.sh", "pbmc" in sc[1])
    queue = [(s, transform(c)) for s, c in sorted(raw, key=order)]

    todo = [(s, c) for s, c in queue if key(c) not in done]
    (print if args.dry_run else log)(f"queue: {len(queue)} commands, {len(queue) - len(todo)} already done, "
        f"{len(todo)} to run (PBMC_MODE={PBMC_MODE})")
    if args.dry_run:
        for s, c in todo:
            print(f"{s}: {c}")
        return

    make_pbmc_subset()
    for i, (script, cmd) in enumerate(todo, 1):
        log(f"START {i}/{len(todo)} [{script}] {cmd}")
        t0 = dt.datetime.now()
        with open(LOGS / "commands.log", "a") as out:
            out.write(f"\n===== {t0:%F %T} {cmd}\n")
            out.flush()
            rc = subprocess.run(cmd, shell=True, cwd=REPO, stdout=out, stderr=subprocess.STDOUT).returncode
        mins = (dt.datetime.now() - t0).total_seconds() / 60
        if rc == 0:
            with open(DONE, "a") as f:
                f.write(key(cmd) + "\n")
            log(f"OK    {i}/{len(todo)} in {mins:.1f} min")
        else:
            with open(FAILED, "a") as f:
                f.write(f"{cmd}\n")
            log(f"FAIL  {i}/{len(todo)} rc={rc} after {mins:.1f} min (see logs/commands.log)")
    log("queue finished")


if __name__ == "__main__":
    sys.exit(main())
