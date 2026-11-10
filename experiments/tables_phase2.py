"""Leakage-free protocol tables:
results/tables/T1_feco_reported_vs_corrected.csv  — FeCo reported → repro (FeCo protocol) → split → corrected FS → +conformal
results/tables/T1b_leakage_selection.csv          — optimism of selecting K (ANOVA removals) / τ on test vs on val
results/tables/T1c_loco_zeroday.csv               — leave-one-category-out zero-day recall (NSL-KDD)
python experiments/tables_phase2.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.select import device_of, get, load_runs  # noqa: E402

FECO = {"central": dict(acc=89.55, recall=86.80, precision=94.39, f1=90.44, fpr=6.82, novel_recall=78.45,
                       DoS=97.16, Probe=96.08, R2L=50.91, U2R=82.50),
        "fl_iid": dict(acc=85.65, recall=81.15, precision=92.73, f1=86.55, fpr=8.41),
        "fl_noniid1": dict(acc=86.23, recall=82.10, precision=92.89, f1=87.16, fpr=8.29),
        "fl_noniid2": dict(acc=85.53, recall=77.90, precision=95.93, f1=85.98, fpr=4.26)}
THR = ["acc", "recall", "precision", "f1", "fpr"]
RNK = ["auroc", "auprc", "tpr@1", "tpr@5"]


CATS = ["DoS", "Probe", "R2L", "U2R"]


def summarize(phase, prefix, thr="feco_p5", split="test"):
    runs = load_runs(phase, prefix, max_seed=5)
    if not runs:
        return None
    vals = {}
    for c, m in runs:
        t = get(m, f"results.template.{split}")
        for k in THR:
            vals.setdefault(k, []).append(100 * t[thr][k])
        for k in RNK:
            vals.setdefault(k, []).append(100 * t["ranking"][k])
        if "novel_recall" in t[thr]:  # FeCo Table V (needs the 43-col test file)
            vals.setdefault("novel_recall", []).append(100 * t[thr]["novel_recall"])
        for k in CATS:
            if k in t[thr].get("recall_by_group", {}):
                vals.setdefault(k, []).append(100 * t[thr]["recall_by_group"][k])
    out = {k: (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0, len(v)) for k, v in vals.items()}
    out["_device"] = device_of(runs)
    return out


def t1():
    rows = []
    spec = {"central": [("phase1", "finalC_full_s", "feco_p5", "repro: FeCo protocol (full train, paper FS, p=5)"),
                        ("phase2", "splitpaperC_s", "feco_p5", "80% split, paper FS, p=5"),
                        ("phase2", "corrC_s", "feco_p5", "corrected: 80% split, FS on split, p=5"),
                        ("phase2", "corrC_s", "conf_a0.05", "corrected + split-conformal α=0.05 (held-out calib)")]}
    for sch in ("iid", "noniid1", "noniid2"):
        spec[f"fl_{sch}"] = [("phase1", f"finalFL_{sch}_full_s", "feco_p5", "repro: FeCo protocol (full train, paper FS, p=5)"),
                             ("phase2", f"splitpaperFL_{sch}_s", "feco_p5", "80% split, paper FS (pooled), p=5"),
                             ("phase2", f"corrFL_{sch}_s", "feco_p5", "corrected: 80% split, federated FS, p=5"),
                             ("phase2", f"corrFL_{sch}_s", "conf_a0.05", "corrected + split-conformal α=0.05 (held-out calib)")]
    for setting, lst in spec.items():
        r0 = {"setting": setting, "protocol": "FeCo reported (TDSC Tab. IV/VII)", "n_seeds": None}
        r0.update({k: f"{v:.2f}" for k, v in FECO[setting].items()})
        rows.append(r0)
        for ph, pre, thr, desc in lst:
            s = summarize(ph, pre, thr)
            if s is None:
                rows.append({"setting": setting, "protocol": desc, "note": "MISSING RUNS"})
                continue
            r = {"setting": setting, "protocol": desc, "n_seeds": s["acc"][2], "device": s["_device"]}
            r.update({k: f"{s[k][0]:.2f}±{s[k][1]:.2f}" for k in THR + RNK + ["novel_recall"] + CATS if k in s})
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/T1_feco_reported_vs_corrected.csv", index=False)
    with open("results/tables/T1_feco_reported_vs_corrected.tex", "w") as f:
        f.write(df[[c for c in ["setting", "protocol"] + THR + ["novel_recall", "auroc", "tpr@1", "device"] if c in df]]
                .fillna("").to_latex(index=False))
    return df


def t1b():
    """For each seed: val-chosen vs test-chosen K / τ, and the test accuracy each implies."""
    rows = []
    for name, opts, fmt in (("anova_K", range(5), "leakK{}_s{}"), ("tau", [0.05, 0.1, 0.2, 0.5], "leakTau{}_s{}")):
        for s in range(5):
            res, devs = [], set()
            for o in opts:
                r = load_runs("phase2", fmt.format(o, s))
                r = [x for x in r if x[0]["run_id"] == fmt.format(o, s)]
                if not r:
                    continue
                devs.add(r[0][1].get("device", "cpu"))
                m = r[0][1]["results"]["template"]
                res.append((o, m["val"]["ranking"]["tpr@1"], m["val"]["ranking"]["auroc"], m["val"]["feco_p5"]["acc"],
                            m["test"]["feco_p5"]["acc"], m["test"]["ranking"]["auroc"]))
            if len(res) < len(list(opts)):
                continue
            v = max(res, key=lambda x: (round(x[1], 4), x[2]))
            v_acc = max(res, key=lambda x: x[3])
            t = max(res, key=lambda x: x[4])
            rows.append({"knob": name, "seed": s, "val_choice(tpr@1)": v[0], "test_acc@val_choice": 100 * v[4],
                         "val_choice(acc)": v_acc[0], "test_acc@val_acc_choice": 100 * v_acc[4],
                         "test_choice(FeCo-style)": t[0], "test_acc@test_choice": 100 * t[4],
                         "optimism_pts": 100 * (t[4] - v[4]), "device": "+".join(sorted(devs)),
                         **{f"test_acc[{o}]": 100 * x[4] for o, *_r in res for x in [next(z for z in res if z[0] == o)]}})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv("results/tables/T1b_leakage_selection.csv", index=False, float_format="%.2f")
    return df


def t1c():
    rows = []
    for c in ("DoS", "Probe", "R2L", "U2R"):
        R = load_runs("phase2", f"loco_{c}_s", max_seed=5)
        if not R:
            continue
        for thr in ("feco_p5", "conf_a0.05"):
            zr = [100 * m["results"]["template"]["zeroday"][thr]["novel_recall"] for _, m in R]
            fp = [100 * m["results"]["template"]["zeroday"][thr]["fpr"] for _, m in R]
            au = [100 * m["results"]["template"]["zeroday"]["ranking"]["auroc"] for _, m in R]
            t1 = [100 * m["results"]["template"]["zeroday"]["ranking"]["tpr@1"] for _, m in R]
            rows.append({"held_out": c, "threshold": thr, "n_seeds": len(R),
                         "zero_day_recall": f"{np.mean(zr):.2f}±{np.std(zr, ddof=1):.2f}",
                         "fpr_val_benign": f"{np.mean(fp):.2f}±{np.std(fp, ddof=1):.2f}",
                         "auroc": f"{np.mean(au):.2f}±{np.std(au, ddof=1):.2f}",
                         "tpr@1": f"{np.mean(t1):.2f}±{np.std(t1, ddof=1):.2f}", "device": device_of(R)})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv("results/tables/T1c_loco_zeroday.csv", index=False)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(t1().to_string())
    print(t1b().to_string())
    print(t1c().to_string())
