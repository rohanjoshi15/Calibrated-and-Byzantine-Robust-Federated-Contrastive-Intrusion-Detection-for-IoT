"""Byzantine analysis (C3).
Outputs: results/tables/T5_whitelist.csv, T5_proto.csv, T5_calib.csv; figure F_byzantine.(png|pdf).
ASR = share of attackers' own botnet flows scored benign (≥ ρ). Clean AUPRC/AUROC on the full test set.
FPR violation: share of clients with FPR > α+1/(n+1) at α=0.05 (test benign, global threshold; exchangeable hold-out, local).
python experiments/tables_phase5.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval.plotstyle import SERIES, save, setup  # noqa: E402
from src.utils.select import load_runs  # noqa: E402


def ms(v):
    v = np.asarray([x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    return (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, len(v)) if len(v) else (np.nan, np.nan, 0)


def whitelist():
    rows = []
    for ds in ("NSL", "NBA"):
        for agg in ("fedavg", "trimmed", "median"):
            specs = [("clean", 0.2, None, f"{ds}_clean_{agg}_s")] + \
                    [("whitelist", f, b, f"{ds}_wl_f{f}_b{b}_{agg}_s") for f in (0.1, 0.2, 0.3) for b in (1.0, 5.0)]
            for att, f, b, pre in specs:
                R = load_runs("phase5", pre, max_seed=5)
                if not R:
                    continue
                rs = [m["fed_results"]["K1_sigma0.0"] for _, m in R]
                row = {"dataset": ds, "agg": agg, "attack": att, "frac": f, "boost": b, "n_seeds": len(rs),
                       "device": "+".join(sorted({m.get("device", "cpu") for _, m in R}))}
                for nm, vals in (("asr_global", [r["byz"].get("asr_global") for r in rs]),
                                 ("asr_robust", [r["byz"].get("asr_robust") for r in rs]),
                                 ("auroc", [r["ranking_test"]["auroc"] for r in rs]),
                                 ("auprc", [r["ranking_test"]["auprc"] for r in rs]),
                                 ("tpr@1", [r["ranking_test"]["tpr@1"] for r in rs]),
                                 ("tpr_conf_global_a0.05", [r["conformal_global_a0.05"]["tpr_mean"] for r in rs]),
                                 ("fpr_viol_test_global_a0.05", [r["conformal_global_a0.05"]["test"]["frac_violate"] for r in rs]),
                                 ("fpr_viol_id_local_a0.05", [r["conformal_local_a0.05"]["id"]["frac_violate"] for r in rs])):
                    mu, sd, _ = ms(vals)
                    row[nm], row[nm + "_std"] = 100 * mu, 100 * sd
                rows.append(row)
    return pd.DataFrame(rows)


def posthoc():
    rp, rc = [], []
    for ds in ("NSL", "NBA"):
        R = load_runs("phase5", f"{ds}_clean_fedavg_s", max_seed=5)
        if not R:
            continue
        dev = "+".join(sorted({m.get("device", "cpu") for _, m in R}))
        keys = sorted(set(k for _, m in R for k in m.get("byz_posthoc", {})))
        for k in keys:
            rs = [m["byz_posthoc"][k] for _, m in R if k in m.get("byz_posthoc", {})]
            att = "proto" if "attackproto" in k else "calib"
            frac = float(k.split("frac")[1].split("_")[0])
            row = {"dataset": ds, "attack": att, "frac": frac, "n_seeds": len(rs), "key": k, "device": dev}
            if att == "proto":
                row["K"] = int(k.split("_K")[1].split("_")[0])
                row["proto_agg"] = k.split("proto_agg")[1] if "proto_agg" in k else "kmeans"
                sup = [r["byz"].get("support") for r in rs if r["byz"].get("support")]
                if sup:
                    row["support_kept_of_total"] = f'{np.mean([x["kept"] for x in sup]):.1f}/{sup[0]["total"]}'
                for nm, vals in (("asr_global", [r["byz"]["asr_global"] for r in rs]),
                                 ("asr_honest_thr", [r["byz"]["asr_honest_oracle"] for r in rs]),
                                 ("fpr_test_local_a0.05", [r["conformal_local_a0.05"]["test"]["mean_fpr"] for r in rs]),
                                 ("fpr_id_local_a0.05", [r["conformal_local_a0.05"]["id"]["mean_fpr"] for r in rs]),
                                 ("auroc", [r["ranking_test"]["auroc"] for r in rs]),
                                 ("auprc", [r["ranking_test"]["auprc"] for r in rs]),
                                 ("tpr@1", [r["ranking_test"]["tpr@1"] for r in rs])):
                    mu, sd, _ = ms(vals)
                    row[nm], row[nm + "_std"] = 100 * mu, 100 * sd
                rp.append(row)
            else:
                for nm, vals in (("rho_global", [r["byz"]["rho_global"] for r in rs]),
                                 ("rho_robust", [r["byz"]["rho_robust"] for r in rs]),
                                 ("rho_honest", [r["byz"]["rho_honest_oracle"] for r in rs]),
                                 ("kept_byz", [r["byz"]["robust_kept_byz"] for r in rs]),
                                 ("n_byz", [r["byz"]["n_byz"] for r in rs])):
                    row[nm] = ms(vals)[0]
                for nm, vals in (("asr_global", [r["byz"]["asr_global"] for r in rs]),
                                 ("asr_robust", [r["byz"]["asr_robust"] for r in rs]),
                                 ("asr_honest_thr", [r["byz"]["asr_honest_oracle"] for r in rs]),
                                 ("tpr_global_a0.05", [r["conformal_global_a0.05"]["tpr_mean"] for r in rs]),
                                 ("tpr_robust_a0.05", [r["robust_global_a0.05"]["tpr_mean"] for r in rs]),
                                 ("testfpr_global_a0.05", [r["conformal_global_a0.05"]["test"]["mean_fpr"] for r in rs]),
                                 ("testfpr_robust_a0.05", [r["robust_global_a0.05"]["test"]["mean_fpr"] for r in rs])):
                    mu, sd, _ = ms(vals)
                    row[nm], row[nm + "_std"] = 100 * mu, 100 * sd
                rc.append(row)
    return pd.DataFrame(rp), pd.DataFrame(rc)


def fig(W, P, C):
    setup()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(8, 6.4), sharey=True)
    for i, ds in enumerate(("NSL", "NBA")):
        ax = axes[i, 0]
        for k, agg in enumerate(("fedavg", "trimmed", "median")):
            g = W[(W.dataset == ds) & (W["agg"] == agg) & (W.attack == "whitelist") & (W.boost == 5.0)].sort_values("frac")
            if len(g):
                ax.errorbar(g.frac, g.asr_global, yerr=g.asr_global_std, color=SERIES[k], marker="o", capsize=2, label=agg)
        ax.set_title(f"{ds}: whitelisting (boost 5)")
        ax = axes[i, 1]
        if len(P):
            lab = {"kmeans": "no defence", "geomed": "geo. median", "trimmed": "trimmed", "support": "client support"}
            for k, (pa, K) in enumerate((("kmeans", 1), ("geomed", 1), ("kmeans", 4), ("geomed", 4), ("support", 4))):
                g = P[(P.dataset == ds) & (P.proto_agg == pa) & (P.K == K)].sort_values("frac")
                if len(g):
                    ax.errorbar(g.frac, g.asr_global, yerr=g.asr_global_std, color=SERIES[k], marker="o", capsize=2,
                                label=f"K={K}, {lab[pa]}")
        ax.set_title(f"{ds}: prototype poisoning")
        ax = axes[i, 2]
        if len(C):
            g = C[C.dataset == ds].sort_values("frac")
            if len(g):
                ax.errorbar(g.frac, g.asr_global, yerr=g.asr_global_std, color=SERIES[0], marker="o", capsize=2, label="global quantile")
                ax.errorbar(g.frac, g.asr_robust, yerr=g.asr_robust_std, color=SERIES[1], marker="o", capsize=2, label="robust quantile")
        ax.set_title(f"{ds}: calibration poisoning")
        axes[i, 0].set_ylabel("attack success rate (%)")
    for ax in axes.flat:
        ax.set_ylim(-3, 103)
        ax.set_xticks([0.1, 0.2, 0.3])
    for ax in axes[1]:
        ax.set_xlabel("malicious client fraction")
    for j in range(3):
        h, l = axes[0, j].get_legend_handles_labels()
        axes[1, j].legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=7, ncol=1)
    fig.tight_layout(h_pad=1.5)
    save(fig, "F_byzantine")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    W = whitelist()
    P, C = posthoc()
    for nm, df in (("T5_whitelist", W), ("T5_proto", P), ("T5_calib", C)):
        if len(df):
            df.to_csv(f"results/tables/{nm}.csv", index=False, float_format="%.3f")
            print(nm)
            print(df.to_string())
    if len(W) or len(P):
        fig(W, P, C)
