"""Select best config among runs by validation metric and store its train block.

python experiments/select.py <phase> <run_prefix> <key> [--det template]
Writes/updates configs/<phase>_selected.json[key] = train cfg of the best run (+ provenance)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.select import best, load_runs, val_key  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("prefix")
    ap.add_argument("key")
    ap.add_argument("--det", default="template")
    ap.add_argument("--field", default="train")
    a = ap.parse_args()
    runs = load_runs(a.phase, a.prefix)
    c, m = best(runs, a.det)
    path = os.path.join("configs", f"{a.phase}_selected.json")
    sel = json.load(open(path)) if os.path.exists(path) else {}
    sel[a.key] = c[a.field]
    sel.setdefault("_provenance", {})[a.key] = {"run_id": c["run_id"], "n_candidates": len(runs),
                                                "val_tpr@1,auroc": val_key(m, a.det)}
    json.dump(sel, open(path, "w"), indent=1)
    print(a.key, "<-", c["run_id"], val_key(m, a.det), "of", len(runs))
