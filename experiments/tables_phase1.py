"""Build results/tables/T_repro.csv (+ .tex): FeCo reported (TDSC Tables IV, V, VII) vs our reproduction.
python experiments/tables_phase1.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.select import device_of, get, load_runs  # noqa: E402

FECO = {  # TDSC Table IV (centralised) and Table VII (FL / self-learning); values in %
    ("central", "-"): dict(acc=89.55, recall=86.80, precision=94.39, f1=90.44, fpr=6.82, novel_recall=78.45,
                           DoS=97.16, Probe=96.08, R2L=50.91, U2R=82.50),
    ("fl", "iid"): dict(acc=85.65, recall=81.15, precision=92.73, f1=86.55, fpr=8.41, DoS=92.69, Probe=98.68, R2L=35.00, U2R=74.00),
    ("fl", "noniid1"): dict(acc=86.23, recall=82.10, precision=92.89, f1=87.16, fpr=8.29, DoS=93.12, Probe=95.67, R2L=41.03, U2R=72.50),
    ("fl", "noniid2"): dict(acc=85.53, recall=77.90, precision=95.93, f1=85.98, fpr=4.26, DoS=83.33, Probe=95.29, R2L=36.49, U2R=76.50),
    ("self", "iid"): dict(acc=81.91, recall=74.53, precision=92.21, f1=82.43, fpr=8.33, DoS=87.66, Probe=92.14, R2L=24.33, U2R=63.07),
    ("self", "noniid1"): dict(acc=82.62, recall=75.69, precision=92.39, f1=83.21, fpr=8.23, DoS=90.48, Probe=87.37, R2L=25.63, U2R=71.81),
    ("self", "noniid2"): dict(acc=67.66, recall=46.51, precision=93.89, f1=62.21, fpr=4.39, DoS=54.20, Probe=47.22, R2L=24.14, U2R=59.25),
}
METRICS = ["acc", "recall", "precision", "f1", "fpr"]
EXTRA = ["auroc", "auprc", "tpr@1", "tpr@5"]
CATS = ["DoS", "Probe", "R2L", "U2R"]
NA = "not computable: test file without sub-class labels"


def collect(prefix):
    """Test metrics at FeCo's p=5 threshold; novel-attack recall (FeCo Table V: attack sub-classes absent from
    KDDTrain+) and per-category recall (FeCo Eq. 10) need the 43-column KDDTest+."""
    runs = load_runs("phase1", prefix, max_seed=5)
    vals = {}
    for c, m in runs:
        t = get(m, "results.template.test")
        for k in METRICS:
            vals.setdefault(k, []).append(100 * t["feco_p5"][k])
        for k in EXTRA:
            vals.setdefault(k, []).append(100 * t["ranking"][k])
        if "novel_recall" in t["feco_p5"]:
            vals.setdefault("novel_recall", []).append(100 * t["feco_p5"]["novel_recall"])
        for k in CATS:
            if k in t["feco_p5"].get("recall_by_group", {}):
                vals.setdefault(k, []).append(100 * t["feco_p5"]["recall_by_group"][k])
    return vals, len(runs), device_of(runs)


def main():
    rows = []
    for (mode, sch), ref in FECO.items():
        prefix = "finalC_full_s" if mode == "central" else f"final{mode.upper()}_{sch}_full_s"
        vals, n, dev = collect(prefix)
        for k in METRICS + EXTRA + ["novel_recall"] + CATS:
            r = {"setting": mode, "partition": sch, "metric": k, "feco_reported": ref.get(k, np.nan), "device": dev}
            if k in vals:
                v = np.array(vals[k])
                r.update(ours_mean=v.mean(), ours_std=v.std(ddof=1) if len(v) > 1 else 0.0, n_seeds=len(v))
                if not np.isnan(r["feco_reported"]):
                    r["gap"] = r["ours_mean"] - r["feco_reported"]
                    r["within_2pts"] = abs(r["gap"]) <= 2.0
            elif k in ref:
                r["note"] = NA
            else:
                continue
            rows.append(r)
    # pre-specified debug attempts, central only, reported as sensitivity (not selection)
    for name in ("A1_literal", "A2_literal_zscore", "A3_sel_log1p"):
        vals, n, dev = collect(f"debugC_{name}_s")
        for k in METRICS + EXTRA + ["novel_recall"] + CATS:
            if k in vals:
                v = np.array(vals[k])
                ref = FECO[("central", "-")].get(k, np.nan)
                rows.append({"setting": f"central_debug_{name}", "partition": "-", "metric": k, "feco_reported": ref,
                             "device": dev,
                             "ours_mean": v.mean(), "ours_std": v.std(ddof=1), "n_seeds": len(v),
                             "gap": v.mean() - ref if not np.isnan(ref) else np.nan,
                             "within_2pts": abs(v.mean() - ref) <= 2 if not np.isnan(ref) else np.nan,
                             "note": "debug attempt D-013 (not used for selection)"})
    df = pd.DataFrame(rows)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/T_repro.csv", index=False, float_format="%.2f")
    piv = df[df.metric.isin(METRICS + ["novel_recall"])].copy()

    def cell(r):
        if pd.isna(r.get("ours_mean")):
            return "--"
        ref = f"{r.feco_reported:.2f}" if pd.notna(r.feco_reported) else "--"
        return f"{r.ours_mean:.2f}$\\pm${r.ours_std:.2f} ({ref})"
    piv["cell"] = piv.apply(cell, axis=1)
    piv = piv.pivot_table(index=["setting", "partition"], columns="metric", values="cell", aggfunc="first")
    piv = piv[[c for c in METRICS + ["novel_recall"] if c in piv.columns]]
    piv["device"] = df.groupby(["setting", "partition"])["device"].first()
    with open("results/tables/T_repro.tex", "w") as f:
        f.write("% ours mean±std over seeds (FeCo reported in parentheses); test = KDDTest+ (43-col file, D-025); "
                "p=5 threshold\n")
        f.write(piv.to_latex(escape=False))
    print(df.to_string())


if __name__ == "__main__":
    main()
