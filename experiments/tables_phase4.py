"""Conformal threshold analysis (C2; H2, H3).
Sources: phase4 nsl_{iid,dir0.1,dir1.0}, phase3 fedNSL_lam0.0 (= Dir 0.5), phase4 nba_dir{0.1,0.5,1.0},
phase3 fedNBA_lam0.0 (= natural). Detector: K=1 template (and K=4 in T4_K4).
Outputs: results/tables/T4_fpr_control.csv (all), T4_fpr_control_perseed.csv, T4_fpr_perclient.csv, T4_fpr_control_K4.csv,
T4_calib_size.csv,
figures F_fpr_vs_alpha.(png|pdf), F_violation_vs_heterogeneity.(png|pdf).
'ref_exact_violate' = expected share of clients flagged as violating if thresholds were EXACT split-conformal under
exchangeability (simulated: F ~ Beta(k, n+1−k), empirical FPR ~ Bin(m, F)/m, violation if > α + 1/(n+1)),
using each run's actual per-client calibration size n and hold-out size m.
python experiments/tables_phase4.py"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval.plotstyle import SERIES, save, setup  # noqa: E402
from src.utils.select import load_runs  # noqa: E402

DEV = {}  # (dataset, partition) -> device labels

SOURCES = [("NSL-KDD", "iid", "phase4", "nsl_iid_s"), ("NSL-KDD", "dir1.0", "phase4", "nsl_dir1.0_s"),
           ("NSL-KDD", "dir0.5", "phase3", "fedNSL_lam0.0_s"), ("NSL-KDD", "dir0.1", "phase4", "nsl_dir0.1_s"),
           ("N-BaIoT", "dir1.0", "phase4", "nba_dir1.0_s"), ("N-BaIoT", "dir0.5", "phase4", "nba_dir0.5_s"),
           ("N-BaIoT", "dir0.1", "phase4", "nba_dir0.1_s"), ("N-BaIoT", "natural", "phase3", "fedNBA_lam0.0_s")]
METHODS = ["feco_global", "feco_local", "conformal_global", "conformal_local", "conformal_local_ceil", "robust_global"]
LABEL = {"feco_global": "FeCo global (train pct.)", "feco_local": "FeCo local (train pct.)",
         "conformal_global": "global fed. quantile", "conformal_local": "per-client conformal",
         "conformal_local_ceil": "per-client conformal (ceil)", "robust_global": "robust quantile"}
ALPHAS = [0.01, 0.05, 0.1]


def ref_violation(ns, ms, alpha, reps=2000, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for n, m in zip(ns, ms):
        if m == 0:
            continue
        k = math.floor(alpha * (n + 1) + 1e-9)
        if k < 1:
            out.append(0.0)
            continue
        F = rng.beta(k, n + 1 - k, reps)
        e = rng.binomial(m, F) / m
        out.append(np.mean(e > alpha + 1.0 / (n + 1)))
    return float(np.mean(out)) if out else np.nan


def id_sizes(c, m):
    r = m["fed_results"]["K1_sigma0.0"]
    if "id_n" in r:
        return r["id_n"]
    from src.fl.fedeval import build_clients
    from src.pipeline import build_data
    D = build_data(c["data"] if c["data"].get("fs") != "federated" else c["data"], c.get("split_seed", c["seed"]))
    C = build_clients(D, c["fl"], c["seed"])
    return [len(p) for p in C["val_benign"]]


def collect(key="K1_sigma0.0"):
    rows, perclient = [], []
    for ds, part, ph, pre in SOURCES:
        R = load_runs(ph, pre, max_seed=5)
        DEV[(ds, part)] = "+".join(sorted({m.get("device", "cpu") for _, m in R}))
        for c, m in R:
            r = m["fed_results"].get(key)
            if r is None:
                continue
            ns, idn = r["calib_n"], id_sizes(c, m)
            for meth in METHODS:
                for a in ALPHAS:
                    nm = f"{meth}_a{a}"
                    if nm not in r:
                        continue
                    x = r[nm]
                    row = {"dataset": ds, "partition": part, "seed": c["seed"], "device": m.get("device", "cpu"),
                           "method": meth, "alpha": a,
                           "tpr_mean": x["tpr_mean"], "trivial_frac": x["trivial_frac"]}
                    for ev in ("id", "test"):
                        for k, v in x[ev].items():
                            row[f"{ev}_{k}"] = v
                    if meth.startswith("conformal_local") and key == "K1_sigma0.0":
                        row["ref_exact_violate_id"] = ref_violation(ns, idn, a, seed=c["seed"])
                    rows.append(row)
                    if key == "K1_sigma0.0":
                        for cid, (f_id, f_te) in enumerate(zip(x["fpr_id_per_client"], x["fpr_test_per_client"])):
                            perclient.append({"dataset": ds, "partition": part, "seed": c["seed"], "device": m.get("device", "cpu"),
                                              "method": meth,
                                              "alpha": a, "client": cid, "fpr_id": f_id, "fpr_test": f_te, "n_calib": ns[cid]})
    return pd.DataFrame(rows), pd.DataFrame(perclient)


def calib_size():
    rows = []
    for ds, part, ph, pre in SOURCES:
        for c, m in load_runs(ph, pre, max_seed=5):
            r = m["fed_results"].get("K1_sigma0.0", {})
            for cap in (25, 50, 100, 200, None):
                for a in ALPHAS:
                    nm = f"conformal_local_cap{cap}_a{a}" if cap else f"conformal_local_a{a}"
                    if nm not in r:
                        continue
                    x = r[nm]
                    rows.append({"dataset": ds, "partition": part, "seed": c["seed"], "device": m.get("device", "cpu"),
                                 "cap": cap or "all", "alpha": a,
                                 "id_mean_abs": x["id"]["mean_abs_fpr_minus_alpha"], "id_frac_violate": x["id"]["frac_violate"],
                                 "id_mean_fpr": x["id"]["mean_fpr"], "tpr_mean": x["tpr_mean"],
                                 "trivial_frac": x["trivial_frac"]})
    return pd.DataFrame(rows)


def figs(T, PC):
    setup()
    import matplotlib.pyplot as plt
    show = ["feco_global", "conformal_global", "conformal_local"]
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.2), sharex=True)
    for i, (ds, part) in enumerate((("NSL-KDD", "dir0.1"), ("N-BaIoT", "natural"))):
        for j, ev in enumerate(("id", "test")):
            ax = axes[i, j]
            for k, meth in enumerate(show):
                g = PC[(PC.dataset == ds) & (PC.partition == part) & (PC.method == meth)]
                if g.empty:
                    continue
                xs, mu, lo, hi = [], [], [], []
                for a in ALPHAS:
                    v = g[g.alpha == a][f"fpr_{ev}"].dropna().values
                    xs.append(a)
                    mu.append(v.mean())
                    lo.append(np.percentile(v, 10))
                    hi.append(np.percentile(v, 90))
                off = (k - 1) * 0.002
                ax.errorbar(np.array(xs) + off, mu, yerr=[np.array(mu) - lo, np.array(hi) - mu], color=SERIES[k],
                            marker="o", capsize=2, label=LABEL[meth])
            ax.plot([0, 0.11], [0, 0.11], color="#9a9893", lw=1, ls="--", zorder=0)
            ax.set_title(f"{ds} ({part}): {'exchangeable hold-out' if ev == 'id' else 'test period (shift/drift)'}")
            if i == 1:
                ax.set_xlabel("target α")
    axes[0, 0].legend(loc="upper left")
    fig.supylabel("per-client FPR (mean; bars = 10th-90th percentile over clients)", fontsize=9)
    fig.tight_layout()
    save(fig, "F_fpr_vs_alpha")
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.8), sharex="col")
    orders = (["iid", "dir1.0", "dir0.5", "dir0.1"], ["dir1.0", "dir0.5", "dir0.1", "natural"])
    for col, (ds, order) in enumerate(zip(("NSL-KDD", "N-BaIoT"), orders)):
        for row, (metric, ylab) in enumerate((("id_frac_violate", "share of clients with\nFPR > α+1/(n+1)"),
                                             ("id_mean_abs_fpr_minus_alpha", "mean over clients\n|FPR − α|"))):
            ax = axes[row, col]
            for k, meth in enumerate(show):
                g = T[(T.dataset == ds) & (T.method == meth) & (T.alpha == 0.05)]
                ys = [g[g.partition == p][metric].mean() for p in order]
                sd = [g[g.partition == p][metric].std(ddof=1) for p in order]
                ax.errorbar(range(len(order)), ys, yerr=sd, color=SERIES[k], marker="o", capsize=2, label=LABEL[meth])
            if row == 0:
                ref = T[(T.dataset == ds) & (T.method == "conformal_local") & (T.alpha == 0.05)]
                ax.plot(range(len(order)), [ref[ref.partition == p]["ref_exact_violate_id"].mean() for p in order],
                        color="#9a9893", ls="--", lw=1, label="exact-conformal reference")
                ax.set_title(ds)
            else:
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels(order)
                ax.set_xlabel("partition (more heterogeneous →)")
            if col == 0:
                ax.set_ylabel(ylab)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("α = 0.05, exchangeable benign hold-out (5 seeds)", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "F_violation_vs_heterogeneity")


def summary(T):
    g = T.groupby(["dataset", "partition", "method", "alpha"])
    cols = ["id_mean_abs_fpr_minus_alpha", "id_frac_violate", "id_mean_fpr", "id_max_fpr", "test_mean_abs_fpr_minus_alpha",
            "test_frac_violate", "test_mean_fpr", "tpr_mean", "trivial_frac", "ref_exact_violate_id"]
    S = g[[c for c in cols if c in T.columns]].agg(["mean", "std"])
    S.columns = [f"{a}_{b}" for a, b in S.columns]
    S = S.reset_index()
    S["n_seeds"] = g.size().values
    S["device"] = g["device"].agg(lambda x: "+".join(sorted(set(x)))).values
    return S


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    T, PC = collect()
    if T.empty:
        sys.exit("no runs yet")
    T.to_csv("results/tables/T4_fpr_control_perseed.csv", index=False, float_format="%.5f")
    PC.to_csv("results/tables/T4_fpr_perclient.csv", index=False, float_format="%.5f")
    S = summary(T)
    S.to_csv("results/tables/T4_fpr_control.csv", index=False, float_format="%.4f")
    T4, _ = collect("K4_sigma0.0")
    if len(T4):
        summary(T4).to_csv("results/tables/T4_fpr_control_K4.csv", index=False, float_format="%.4f")
    CS = calib_size()
    CS["n_seeds"] = 1
    CS.groupby(["dataset", "partition", "cap", "alpha"]).agg(
        {**{k: "mean" for k in ("seed", "id_mean_abs", "id_frac_violate", "id_mean_fpr", "tpr_mean", "trivial_frac")},
         "n_seeds": "sum", "device": lambda d: "+".join(sorted(set(d)))}).drop(columns="seed").reset_index().to_csv(
        "results/tables/T4_calib_size.csv", index=False, float_format="%.4f")
    a5 = S[S.alpha == 0.05]
    print(a5[["dataset", "partition", "method", "id_mean_abs_fpr_minus_alpha_mean", "id_frac_violate_mean",
              "ref_exact_violate_id_mean", "test_mean_fpr_mean", "test_frac_violate_mean", "tpr_mean_mean"]].to_string())
    figs(T, PC)
