"""H1: paired comparison K=1 vs VAL-SELECTED multi-prototype detector, test TPR@1%FPR.
Settings (seeds 0-9, all cuda): NSL-KDD central (phase3 cenNSL_lam0.0), NSL-KDD federated (phase3 fedNSL_lam0.0,
Dir 0.5), CICIoT2023 central (phase6 cen_CIC_feco), CICIoT2023 federated (phase8 fedCIC, Dir 0.5 attacks).
Candidates (identical across settings): K=1 template; spherical k-means max-cos K ∈ {2,4,8,16}; vMF mixture K ∈ {4,8}
(federated: server prototypes + one federated vMF round). vMF K=1 is rank-equivalent to the template and excluded.
Selection per seed by val TPR@1% (rounded 4 d.p.), tie-break val AUROC; never test.
Reports: mean paired difference (pts), Wilcoxon signed-rank two-sided p (zero differences dropped), bootstrap 95% CI
of the mean difference (10,000 resamples of seeds, percentile), wins/ties/losses, selection counts.
Fixed-K comparisons are written ONLY to a sensitivity table (post-hoc; never a 'best' configuration).
Outputs: results/tables/T_H1_valselected.csv, T_H1_valselected_perseed.csv, T_H1_fixedK_sensitivity.csv.
python experiments/tables_h1.py"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.select import device_of, load_runs  # noqa: E402

CENTRAL = ["template", "kmeans2", "kmeans4", "kmeans8", "kmeans16", "vmf4", "vmf8"]
FED = ["K1_sigma0.0", "K2_sigma0.0", "K4_sigma0.0", "K8_sigma0.0", "K16_sigma0.0", "vmfK4", "vmfK8"]
LABEL = dict(zip(FED, CENTRAL))
SETTINGS = [("NSL-KDD", "central", "phase3", "cenNSL_lam0.0_s"), ("NSL-KDD", "federated", "phase3", "fedNSL_lam0.0_s"),
            ("CICIoT2023", "central", "phase6", "cen_CIC_feco_s"), ("CICIoT2023", "federated", "phase8", "fedCIC_s")]


def detectors(m, fed):
    """{label: (val ranking, test ranking)}"""
    if fed:
        fr = m["fed_results"]
        return {LABEL[k]: (fr[k]["ranking_val"], fr[k]["ranking_test"]) for k in FED if k in fr}
    r = m["results"]
    return {k: (r[k]["val"]["ranking"], r[k]["test"]["ranking"]) for k in CENTRAL if k in r}


def select(dets):
    return max(dets, key=lambda d: (round(dets[d][0]["tpr@1"], 4), dets[d][0]["auroc"]))


def boot_ci(d, reps=10000, seed=0):
    rng = np.random.RandomState(seed)
    mu = d[rng.randint(0, len(d), (reps, len(d)))].mean(1)
    return float(np.percentile(mu, 2.5)), float(np.percentile(mu, 97.5))


def paired(d):
    d = np.asarray(d, float)
    nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    lo, hi = boot_ci(d)
    return {"mean_diff_pts": d.mean(), "std_diff": d.std(ddof=1), "ci95_lo": lo, "ci95_hi": hi, "wilcoxon_p": p,
            "n_nonzero": len(nz), "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum())}


def main():
    per, summ, sens = [], [], []
    for ds, mode, ph, pre in SETTINGS:
        R = load_runs(ph, pre)
        if not R:
            print("missing", ds, mode)
            continue
        fed = mode == "federated"
        rows = []
        for c, m in sorted(R, key=lambda x: x[0]["seed"]):
            dets = detectors(m, fed)
            sel = select(dets)
            row = {"dataset": ds, "mode": mode, "seed": c["seed"], "device": m.get("device", "cpu"),
                   "val_selected": sel, "tpr1_K1": 100 * dets["template"][1]["tpr@1"],
                   "tpr1_selected": 100 * dets[sel][1]["tpr@1"],
                   "val_tpr1_K1": 100 * dets["template"][0]["tpr@1"], "val_tpr1_selected": 100 * dets[sel][0]["tpr@1"]}
            for d in CENTRAL[1:]:
                if d in dets:
                    row[f"tpr1_{d}"] = 100 * dets[d][1]["tpr@1"]
            rows.append(row)
        P = pd.DataFrame(rows)
        per.append(P)
        s = {"dataset": ds, "mode": mode, "n_seeds": len(P), "device": device_of(R),
             "tpr1_K1_mean": P.tpr1_K1.mean(), "tpr1_K1_std": P.tpr1_K1.std(ddof=1),
             "tpr1_selected_mean": P.tpr1_selected.mean(), "tpr1_selected_std": P.tpr1_selected.std(ddof=1),
             **paired(P.tpr1_selected - P.tpr1_K1),
             "selected_counts": ";".join(f"{k}:{v}" for k, v in P.val_selected.value_counts().items())}
        summ.append(s)
        for d in CENTRAL[1:]:
            if f"tpr1_{d}" in P:
                sens.append({"dataset": ds, "mode": mode, "fixed_detector": d, "n_seeds": len(P), "device": device_of(R),
                             "tpr1_mean": P[f"tpr1_{d}"].mean(), "tpr1_std": P[f"tpr1_{d}"].std(ddof=1),
                             **paired(P[f"tpr1_{d}"] - P.tpr1_K1),
                             "note": "sensitivity only: fixed K chosen post hoc, not a selected configuration"})
    os.makedirs("results/tables", exist_ok=True)
    S = pd.DataFrame(summ)
    S.to_csv("results/tables/T_H1_valselected.csv", index=False, float_format="%.4f")
    pd.concat(per).to_csv("results/tables/T_H1_valselected_perseed.csv", index=False, float_format="%.3f")
    pd.DataFrame(sens).to_csv("results/tables/T_H1_fixedK_sensitivity.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(S.to_string())


if __name__ == "__main__":
    main()
