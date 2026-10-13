"""Like SCENE-reproducibility/scripts/render_notebooks.py, but keeps going after a failing cell.

SIMBA and TCR were not run in this reproduction, so cells that need them fail; everything else
in the notebook still runs and saves its figures. Prints which cells failed and why.
Usage (from reproduction/): .venv/bin/python render_allow_errors.py NOTEBOOK.ipynb ...
"""
import argparse
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parent / "SCENE-reproducibility"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("notebooks", type=Path, nargs="+")
    p.add_argument("--timeout", type=int, default=3600)
    args = p.parse_args()
    output = REPO / "tmp" / "executed"
    output.mkdir(parents=True, exist_ok=True)
    for path in args.notebooks:
        path = path.resolve()
        nb = nbformat.read(path, as_version=4)
        km = KernelManager(kernel_name="python3")
        km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(nb, km=km, timeout=args.timeout, allow_errors=True,
                                resources={"metadata": {"path": str(REPO)}})
        print(f"Executing {path.name}", flush=True)
        try:
            client.execute()
        finally:
            nbformat.write(nb, output / path.name)
            if km.has_kernel:
                km.shutdown_kernel(now=True)
        errors = [(i, o.get("ename"), str(o.get("evalue"))[:150])
                  for i, c in enumerate(nb.cells) if c.cell_type == "code"
                  for o in c.get("outputs", []) if o.get("output_type") == "error"]
        n_code = sum(c.cell_type == "code" for c in nb.cells)
        print(f"Completed {path.name}: {n_code - len(errors)}/{n_code} code cells ok", flush=True)
        for i, ename, evalue in errors:
            print(f"   cell {i}: {ename}: {evalue}", flush=True)


if __name__ == "__main__":
    main()
