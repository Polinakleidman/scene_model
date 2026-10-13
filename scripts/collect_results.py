"""Collect all results/*/metrics.json into tidy CSV tables in results_tables/:

  full.csv            one row per full SCENE run (held-out metrics, clustering, protein-gene summary)
  baseline_full.csv   scVI / totalVI clustering
  lodo.csv            one row per (SCENE run or baseline, label level, embedding method): transfer acc / F1
  protein_pairs.csv   one row per (run, protein->gene pair): percentile rank of the own gene
"""
import json
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]


def parse_scene_id(rid):
    # e.g. pbmc_full_rna+prot_zip_decoupled_d16_s0[_P3]
    p = rid.split("_")
    ds, mode, mod = p[0], p[1], p[2]
    i = next(k for k, s in enumerate(p) if s.startswith("d") and s[1:].isdigit())
    return {"dataset": ds, "mode": mode, "modality": mod, "likelihood": "_".join(p[3:i]),
            "dim": int(p[i][1:]), "seed": int(p[i + 1][1:]), "donor": p[i + 2] if len(p) > i + 2 else None}


def main():
    full, base_full, lodo, pairs = [], [], [], []
    for f in sorted((PROJ / "results").glob("*/metrics.json")):
        m = json.loads(f.read_text())
        rid = m["run_id"]
        if "method" in m:   # baseline
            meta = {"dataset": m["dataset"], "model": m["method"], "mode": m["mode"],
                    "donor": rid.split("_")[-1] if m["mode"] == "lodo" else None}
            if m["mode"] == "full":
                for lab, c in m["clustering"].items():
                    base_full.append({**meta, "seed": int(rid.split("_s")[-1]), "label": lab, **c})
            else:
                for lab, t in m["transfer"].items():
                    lodo.append({**meta, "label": lab, "embedding": "scArches", **t["scarches"]})
            continue
        meta = parse_scene_id(rid)
        if meta["mode"] == "full":
            row = {**meta, "run_id": rid, "train_time_s": m["heldout"].get("train_time_s")}
            for k, v in m["heldout"].get("rna", {}).items():
                row[f"rna_{k}"] = v
            for k, v in m["heldout"].get("protein", {}).items():
                row[f"prot_{k}"] = v
            for lab, c in m["clustering"].items():
                for k, v in c.items():
                    row[f"{lab}_{k}"] = v
            pg = m.get("protein_gene")
            if pg:
                row.update({"pg_n_pairs": pg["n_pairs"], "pg_median_rank": pg["median_rank_pct"],
                            "pg_top1": pg["frac_top1pct"], "pg_top5": pg["frac_top5pct"]})
                for pair, r in pg["per_pair"].items():
                    prot, gene = pair.split("->")
                    pairs.append({**meta, "protein": prot, "gene": gene, "rank_pct": r})
            full.append(row)
        else:
            model = f"SCENE {meta['modality']} {meta['likelihood']}"
            for lab, t in m["transfer"].items():
                for emb, v in t.items():
                    lodo.append({"dataset": meta["dataset"], "model": model, "mode": "lodo", "donor": meta["donor"],
                                 "label": lab, "embedding": emb, **v, "foldin_time_s": m.get("foldin_time_s")})
    out = PROJ / "results_tables"
    out.mkdir(exist_ok=True)
    for name, rows in [("full", full), ("baseline_full", base_full), ("lodo", lodo), ("protein_pairs", pairs)]:
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(rows)} rows")


if __name__ == "__main__":
    main()
