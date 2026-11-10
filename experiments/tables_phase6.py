"""Full-evaluation analysis. Outputs (results/tables): T6_central.csv, T6_fl.csv, T6_lofo.csv, T6_perdevice.csv.
For FeCo/CLAD, 'K=1' = template; 'MP' = val-selected K (by val TPR@1%) among K∈{1,2,4,8,16} per seed.
Threshold metrics at split-conformal α=0.05 (pooled held-out calib) and FeCo p=5. 5 seeds.
python experiments/tables_phase6.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.select import load_runs  # noqa: E402

DS = ("NSL", "NBA", "CIC")


def pick(res, mode):
    if mode == "K1":
        return "template" if "template" in res else "native"
    cands = [k for k in res if k.startswith(("template", "kmeans", "vmf"))]
    return max(cands, key=lambda d: (round(res[d]["val"]["ranking"]["tpr@1"], 4), res[d]["val"]["ranking"]["auroc"]))


def metrics_of(r):
    t = r["test"]
    out = {k: 100 * t["ranking"][k] for k in ("auroc", "auprc", "tpr@1", "tpr@5")}
    for thr in ("conf_a0.05", "feco_p5"):
        for k in ("acc", "recall", "precision", "f1", "fpr"):
            out[f"{k}@{thr}"] = 100 * t[thr][k]
        if "novel_recall" in t[thr]:
            out[f"novel_recall@{thr}"] = 100 * t[thr]["novel_recall"]
    return out


def agg(rows, keys):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    num = [c for c in df.columns if c not in keys + ["seed", "detector", "device"]]
    g = df.groupby(keys)
    out = g[num].mean().add_suffix("")
    sd = g[num].std(ddof=1).add_suffix("_std")
    n = g.size().rename("n_seeds")
    det = g["detector"].agg(lambda x: ",".join(sorted(set(x)))) if "detector" in df else None
    dev = g["device"].agg(lambda x: "+".join(sorted(set(x)))) if "device" in df else None
    res = pd.concat([out, sd, n] + [x for x in (det, dev) if x is not None], axis=1).reset_index()
    return res


def central():
    rows = []
    for ds in DS:
        for meth in ("feco", "clad", "svdd", "sad", "ae", "iforest"):
            for c, m in load_runs("phase6", f"cen_{ds}_{meth}_s", max_seed=5):
                res = m["results"]
                for mode in (("K1", "MP") if meth in ("feco", "clad") else ("K1",)):
                    d = pick(res, mode)
                    name = meth if mode == "K1" else f"{meth}-MP"
                    rows.append({"dataset": ds, "method": name, "seed": c["seed"], "detector": d,
                                 "device": m.get("device", "cpu"), **metrics_of(res[d])})
    return agg(rows, ["dataset", "method"])


def fl():
    rows = []
    for c, m in load_runs("phase6", "fl_", max_seed=5):
        _, ds, part, meth, algo, _s = c["run_id"].split("_")
        res = m["results"]
        for mode in (("K1", "MP") if meth in ("feco", "clad") else ("K1",)):
            d = pick(res, mode)
            rows.append({"dataset": ds, "partition": part, "method": meth if mode == "K1" else f"{meth}-MP",
                         "algo": algo, "seed": c["seed"], "detector": d, "device": m.get("device", "cpu"),
                         **metrics_of(res[d])})
    return agg(rows, ["dataset", "partition", "method", "algo"])


def lofo():
    rows = []
    for c, m in load_runs("phase6", "lofo_CIC_", max_seed=5):
        fam = c["data"]["holdout"]
        res = m["results"]
        for mode in ("K1", "MP"):
            d = pick(res, mode)
            r = metrics_of(res[d])
            rows.append({"held_out": fam, "detector_mode": mode, "seed": c["seed"], "detector": d,
                         "device": m.get("device", "cpu"), **r})
    return agg(rows, ["held_out", "detector_mode"])


def perdevice():
    rows = []
    for c, m in load_runs("phase6", "dev_", max_seed=5):
        dev = c["data"]["devices"][0]
        r = m["results"]["template"]["test"]
        rows.append({"nbaiot_device": dev, "seed": c["seed"], "detector": "template", "device": m.get("device", "cpu"),
                     "recall@p5": 100 * r["feco_p5"]["recall"], "fpr@p5": 100 * r["feco_p5"]["fpr"],
                     "acc@p5": 100 * r["feco_p5"]["acc"], "recall@conf0.05": 100 * r["conf_a0.05"]["recall"],
                     "fpr@conf0.05": 100 * r["conf_a0.05"]["fpr"], "auroc": 100 * r["ranking"]["auroc"],
                     "tpr@1": 100 * r["ranking"]["tpr@1"],
                     "fpr@tpr99.98": 100 * r["ranking"].get("fpr@tpr99.98", np.nan)})
    return agg(rows, ["nbaiot_device"])


def fig_zeroday():
    """Bar chart: recall on a held-out (never-trained) family/category at conformal α=0.05."""
    from src.eval.plotstyle import SERIES, save, setup
    import matplotlib.pyplot as plt
    setup()
    L = pd.read_csv("results/tables/T6_lofo.csv") if os.path.exists("results/tables/T6_lofo.csv") else None
    N = pd.read_csv("results/tables/T1c_loco_zeroday.csv") if os.path.exists("results/tables/T1c_loco_zeroday.csv") else None
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), gridspec_kw={"width_ratios": [7, 4]})
    if L is not None:
        g = L[L.detector_mode == "K1"].sort_values("novel_recall@conf_a0.05", ascending=False)
        axes[0].bar(g.held_out, g["novel_recall@conf_a0.05"], yerr=g.get("novel_recall@conf_a0.05_std"), color=SERIES[0],
                    width=0.6, capsize=2)
        axes[0].set_title("CICIoT2023: family held out of training")
        axes[0].set_ylabel("zero-day recall (%)\nconformal α=0.05")
        axes[0].tick_params(axis="x", rotation=30)
    if N is not None:
        g = N[N.threshold == "conf_a0.05"]
        mu = g.zero_day_recall.str.split("±").str[0].astype(float)
        sd = g.zero_day_recall.str.split("±").str[1].astype(float)
        axes[1].bar(g.held_out, mu, yerr=sd, color=SERIES[1], width=0.6, capsize=2)
        axes[1].set_title("NSL-KDD: category held out (LOCO)")
    for ax in axes:
        ax.set_ylim(0, 105)
    fig.tight_layout()
    save(fig, "F_zeroday_lofo")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for nm, f in (("T6_central", central), ("T6_fl", fl), ("T6_lofo", lofo), ("T6_perdevice", perdevice)):
        df = f()
        if len(df):
            df.to_csv(f"results/tables/{nm}.csv", index=False, float_format="%.3f")
            show = [c for c in df.columns if not c.endswith("_std") and ("@feco" not in c)][:14]
            print(nm)
            print(df[show].to_string())
    fig_zeroday()
