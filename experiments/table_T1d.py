"""T1d: per-category recall on the NSL-KDD validation split (seen attack types only) for the corrected
central runs (phase2 corrC_s*), template detector, thresholds split-conformal α=0.05 and FeCo p=5.
Shows that val recall is saturated (98–100%) and is not a zero-day proxy (compare T1c LOCO).
Output: results/tables/T1d_val_per_category_recall.csv.  python experiments/table_T1d.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.select import load_runs  # noqa: E402


def main(prefix="corrC_s"):
    R = [(c, m) for c, m in load_runs("phase2", prefix) if c["seed"] < 5]
    rows = []
    for thr in ("conf_a0.05", "feco_p5"):
        row = {"threshold": thr, "n_seeds": len(R), "device": ",".join(sorted({m.get("device", "cpu") for _, m in R}))}
        for cat in ("DoS", "Probe", "R2L", "U2R"):
            v = np.array([100 * m["results"]["template"]["val"][thr]["recall_by_group"][cat] for _, m in R])
            row[f"{cat}_mean"], row[f"{cat}_std"] = v.mean(), v.std(ddof=1)
        f = np.array([100 * m["results"]["template"]["val"][thr]["fpr"] for _, m in R])
        row["val_fpr_mean"], row["val_fpr_std"] = f.mean(), f.std(ddof=1)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/T1d_val_per_category_recall.csv", index=False, float_format="%.2f")
    print(df.to_string())


if __name__ == "__main__":
    main()
