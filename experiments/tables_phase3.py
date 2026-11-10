"""Multi-prototype analysis (C1).
Outputs: results/tables/T3_central_K.csv, T3_fed_K.csv, T3_privacy_sigma.csv (descriptive, seeds 0-4);
figures F_K_tpr1.(png|pdf), F_privacy_utility.(png|pdf).
H1 is NOT tested here: the only H1 test is experiments/tables_h1.py (val-selected, seeds 0-9).
python experiments/tables_phase3.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval.plotstyle import SERIES, save, setup  # noqa: E402
from src.utils.select import device_of, load_runs  # noqa: E402

DETS = ["template", "kmeans2", "kmeans4", "kmeans8", "kmeans16", "vmf1", "vmf4", "vmf8"]
KS = [1, 2, 4, 8, 16]


def central():
    rows, paired = [], []
    for ds in ("NSL", "NBA"):
        for lam in (0.0, 0.1, 0.5):
            R = load_runs("phase3", f"cen{ds}_lam{lam}_s", max_seed=5)
            if not R:
                continue
            for det in DETS:
                v = [m["results"][det] for _, m in R if det in m["results"]]
                if not v:
                    continue
                r = {"dataset": ds, "lam": lam, "detector": det, "n_seeds": len(v), "device": device_of(R)}
                for k in ("tpr@1", "tpr@5", "auroc", "auprc"):
                    a = np.array([100 * x["test"]["ranking"][k] for x in v])
                    r[k] = a.mean()
                    r[k + "_std"] = a.std(ddof=1)
                    r["val_" + k] = np.mean([100 * x["val"]["ranking"][k] for x in v])
                rows.append(r)
            # H1 paired: per seed val-selected detector (among kmeans K & template) vs template
            for c, m in R:
                res = m["results"]
                cands = [d for d in DETS if d in res]  # template, k-means max-cos, vMF mixtures
                best = max(cands, key=lambda d: (round(res[d]["val"]["ranking"]["tpr@1"], 4), res[d]["val"]["ranking"]["auroc"]))
                row = {"dataset": ds, "lam": lam, "seed": c["seed"], "device": m.get("device", "cpu"), "val_selected": best,
                       "tpr1_K1": 100 * res["template"]["test"]["ranking"]["tpr@1"],
                       "tpr1_sel": 100 * res[best]["test"]["ranking"]["tpr@1"]}
                for K in KS[1:]:
                    row[f"tpr1_K{K}"] = 100 * res[f"kmeans{K}"]["test"]["ranking"]["tpr@1"]
                for K in (4, 8):
                    row[f"tpr1_vmf{K}"] = 100 * res[f"vmf{K}"]["test"]["ranking"]["tpr@1"]
                paired.append(row)
    return pd.DataFrame(rows), pd.DataFrame(paired)


def fed():
    rows = []
    for ds in ("NSL", "NBA"):
        for lam in (0.0, 0.1):
            R = load_runs("phase3", f"fed{ds}_lam{lam}_s", max_seed=5)
            for K in KS:
                for sig in (0.0, 0.01, 0.05, 0.1):
                    key = f"K{K}_sigma{sig}"
                    v = [m["fed_results"][key] for _, m in R if key in m["fed_results"]]
                    if not v:
                        continue
                    r = {"dataset": ds, "lam": lam, "K": K, "sigma": sig, "n_seeds": len(v), "device": device_of(R)}
                    for k in ("tpr@1", "tpr@5", "auroc", "auprc"):
                        a = np.array([100 * x["ranking_test"][k] for x in v])
                        r[k], r[k + "_std"] = a.mean(), a.std(ddof=1)
                    r["tpr_conf_local_a0.05"] = np.mean([100 * x["conformal_local_a0.05"]["tpr_mean"] for x in v])
                    rows.append(r)
            for K in KS:
                v = [m["fed_results"][f"vmfK{K}"] for _, m in R if f"vmfK{K}" in m["fed_results"]]
                if not v:
                    continue
                r = {"dataset": ds, "lam": lam, "K": K, "sigma": "vmf", "n_seeds": len(v), "device": device_of(R)}
                for k in ("tpr@1", "tpr@5", "auroc", "auprc"):
                    a = np.array([100 * x["ranking_test"][k] for x in v])
                    r[k], r[k + "_std"] = a.mean(), a.std(ddof=1)
                r["tpr_conf_local_a0.05"] = np.mean([100 * x["conformal_local_a0.05"]["tpr_mean"] for x in v])
                rows.append(r)
    S = pd.DataFrame(rows)
    F = S[S.sigma.isin([0.0, "vmf"])].copy() if len(S) else S
    return F, S


def figs(C, F, S):
    setup()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=False)
    for ax, ds, title in zip(axes, ("NSL", "NBA"), ("NSL-KDD (KDDTest+)", "N-BaIoT (cross-device)")):
        i = 0
        for src, df in (("central", C), ("federated", F)):
            if df is None or df.empty:
                continue
            for lam in sorted(df[df.dataset == ds].lam.unique()):
                if src == "central":
                    g = df[(df.dataset == ds) & (df.lam == lam) & df.detector.str.match(r"^(template|kmeans)")]
                    xs = [1 if d == "template" else int(d[6:]) for d in g.detector]
                else:
                    g = df[(df.dataset == ds) & (df.lam == lam) & (df.sigma == 0.0)]
                    xs = list(g.K)
                if g.empty:
                    continue
                o = np.argsort(xs)
                ax.errorbar(np.array(xs)[o], g["tpr@1"].values[o], yerr=g["tpr@1_std"].values[o], color=SERIES[i],
                            marker="o", capsize=2, label=f"{src}, λ={lam}")
                i += 1
        ax.set_xscale("log", base=2)
        ax.set_xticks(KS)
        ax.set_xticklabels(KS)
        ax.set_xlabel("number of prototypes K")
        ax.set_title(title)
    axes[0].set_ylabel("test TPR @ 1% FPR (%)\n(max-cos k-means prototypes)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "F_K_tpr1")
    if S is not None and not S.empty:
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
        for ax, ds, title in zip(axes, ("NSL", "NBA"), ("NSL-KDD", "N-BaIoT")):
            g0 = S[(S.dataset == ds) & (S.lam == 0.0) & (S.sigma != "vmf")].copy()
            g0["sigma"] = g0.sigma.astype(float)
            for i, K in enumerate(KS):
                g = g0[g0.K == K].sort_values("sigma")
                if g.empty:
                    continue
                ax.errorbar(g.sigma, g["tpr@1"], yerr=g["tpr@1_std"], color=SERIES[i], marker="o", capsize=2, label=f"K={K}")
            ax.set_xlabel("Gaussian noise σ on client centroids")
            ax.set_title(title)
        axes[0].set_ylabel("test TPR @ 1% FPR (%)")
        axes[1].legend(loc="best", ncol=2)
        save(fig, "F_privacy_utility")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    C, P = central()
    os.makedirs("results/tables", exist_ok=True)
    if len(C):
        C.to_csv("results/tables/T3_central_K.csv", index=False, float_format="%.3f")
        print(C[["dataset", "lam", "detector", "tpr@1", "tpr@1_std", "auroc", "val_tpr@1"]].to_string())
    F, S = fed()
    if len(F):
        F.to_csv("results/tables/T3_fed_K.csv", index=False, float_format="%.3f")
        S.to_csv("results/tables/T3_privacy_sigma.csv", index=False, float_format="%.3f")
        print(F.to_string())
    figs(C if len(C) else None, F if len(F) else None, S if len(S) else None)
