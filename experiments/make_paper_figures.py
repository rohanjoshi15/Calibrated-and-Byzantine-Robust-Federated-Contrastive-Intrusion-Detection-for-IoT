"""Paper figures -> paper_assets/figures/*.pdf (+ .png preview).

Data: results/tables/*.csv and, for the exact-conformal reference, raw results/**/metrics.json via
experiments/make_paper_tables.ref_exact. No experiment is re-run; results/ is read-only.
Style: Liberation Serif (metric-compatible with Times New Roman; STIX math) at 8 pt, TrueType fonts embedded,
Okabe-Ito palette with one fixed colour per method in every figure, widths 3.5 in (single) / 7.16 in (double).

python experiments/make_paper_figures.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
os.chdir(ROOT)
import make_paper_tables as MT  # noqa: E402  (ref_exact, P4SRC; importing runs no table code)

TAB = "results/tables"
OUT = "paper_assets/figures"
W1, W2 = 3.5, 7.16
H1, H2 = 2.2, 2.4

# Okabe-Ito; one colour per method, identical in every figure
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
COL = {"feco_global": OI["vermillion"], "conformal_global": OI["orange"], "conformal_local": OI["blue"],
       "robust": OI["green"], "nodef": OI["purple"], "kmeans": OI["black"], "vmf": OI["sky"], "ref": "#7F7F7F"}
LAB = {"feco_global": "FeCo global", "conformal_global": "Global quantile", "conformal_local": "Per-client conformal"}
METH3 = ["feco_global", "conformal_global", "conformal_local"]


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "Times", "Nimbus Roman"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.handlelength": 1.8, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.25,
        "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.5, "axes.axisbelow": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150,
        "errorbar.capsize": 1.5,
    })


def panel(ax, k, x=-0.2, y=1.04):
    ax.text(x, y, f"({'abcdefgh'[k]})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    w, h = fig.get_size_inches()
    fig.savefig(os.path.join(OUT, name + ".pdf"))  # fixed size: no bbox_inches='tight' (keeps the stated width)
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    print(f"wrote {OUT}/{name}.pdf  {w:.2f} x {h:.2f} in")


def csv(n):
    return pd.read_csv(os.path.join(TAB, n + ".csv"))


def perclient(ds, part, a=0.05):
    """Per-client ID and test FPR, all 5 seeds pooled; client set as fedeval (clients with test-benign data)."""
    PC = csv("T4_fpr_perclient")
    g = PC[(PC.dataset == ds) & (PC.partition == part) & (PC.alpha == a)]
    return g[g.fpr_test.notna() & g.fpr_id.notna()]


def t4():
    return csv("T4_fpr_control")


def f_teaser():
    T = t4()
    g = perclient("N-BaIoT", "natural")
    runs = MT.load_runs("phase3", "fedNBA_lam0.0_s", max_seed=5)
    ref = MT.ref_exact(runs, 0.05)
    fig, axs = plt.subplots(1, 3, figsize=(W2, H1), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    bins = np.linspace(0, 0.20, 33)
    ymax = 0
    for k, meth in enumerate(("feco_global", "conformal_local")):
        ax = axs[k]
        x = g[g.method == meth]
        viol = np.mean(x.fpr_id.values > 0.05 + 1 / (x.n_calib.values + 1))
        row = T[(T.dataset == "N-BaIoT") & (T.partition == "natural") & (T.method == meth) & (T.alpha == 0.05)].iloc[0]
        assert abs(viol - row.id_frac_violate_mean) < 0.002, (meth, viol, row.id_frac_violate_mean)  # pooled = seed mean here
        n, _, _ = ax.hist(x.fpr_id.values, bins=bins, color=COL[meth], alpha=0.85, edgecolor="white", linewidth=0.3,
                          label=LAB[meth])
        ymax = max(ymax, n.max())
        ax.axvline(0.05, color="black", lw=1.0, ls="--")
        ax.set_xlabel("Per-client FPR (exchangeable hold-out)")
        txt = f"{LAB[meth]}\nviolating: {100 * row.id_frac_violate_mean:.1f}%"
        if meth == "conformal_local":
            txt += f"\nexact ref.: {100 * ref:.1f}%"
        ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.text(0.052, 0.02, r"$\alpha$", transform=ax.get_xaxis_transform(), fontsize=8, ha="left", va="bottom")
        panel(ax, k)
    for ax in axs[:2]:
        ax.set_ylim(0, ymax * 1.6)
        ax.set_xlim(0, 0.20)
    axs[0].set_ylabel("Clients (5 seeds pooled)")
    # (c) calibration poisoning, f = 0.3
    C = csv("T5_calib")
    ax = axs[2]
    xs = np.arange(2)
    wdt = 0.36
    for j, (col, key, lab) in enumerate(((COL["nodef"], "asr_global", "Global quantile (no defence)"),
                                         (COL["robust"], "asr_robust", "Robust quantile"))):
        vals = [C[(C.dataset == d) & (C.frac == 0.3)].iloc[0][key] for d in ("NSL", "NBA")]
        sds = [C[(C.dataset == d) & (C.frac == 0.3)].iloc[0][key + "_std"] for d in ("NSL", "NBA")]
        b = ax.bar(xs + (j - 0.5) * wdt, vals, wdt, yerr=sds, color=col, label=lab, ecolor="black",
                   error_kw={"elinewidth": 0.6})
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 2, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["NSL-KDD", "N-BaIoT"])
    ax.set_ylabel("Attack success rate (%)")
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(r"Calibration poisoning, $f=0.3$")
    panel(ax, 2)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.9, wspace=0.32)
    save(fig, "F_teaser")


def f_fpr_vs_alpha():
    PC = csv("T4_fpr_perclient")
    alphas = [0.01, 0.05, 0.1]
    fig, axs = plt.subplots(1, 4, figsize=(W2, H2))
    spec = [("NSL-KDD", "dir0.1", "id", "NSL-KDD Dir(0.1)\nexchangeable hold-out"),
            ("NSL-KDD", "dir0.1", "test", "NSL-KDD Dir(0.1)\nKDDTest+ (shift)"),
            ("N-BaIoT", "natural", "id", "N-BaIoT natural\nexchangeable hold-out"),
            ("N-BaIoT", "natural", "test", "N-BaIoT natural\ntest period (drift)")]
    for k, (ds, part, ev, lab) in enumerate(spec):
        ax = axs[k]
        for i, meth in enumerate(METH3):
            mu, lo, hi = [], [], []
            for a in alphas:
                g = PC[(PC.dataset == ds) & (PC.partition == part) & (PC.method == meth) & (PC.alpha == a)]
                g = g[g.fpr_test.notna() & g.fpr_id.notna()]
                v = g[f"fpr_{ev}"].values
                mu.append(v.mean())
                lo.append(np.percentile(v, 10))
                hi.append(np.percentile(v, 90))
            mu, lo, hi = map(np.array, (mu, lo, hi))
            xs = np.array(alphas) + (i - 1) * 0.0025
            ax.errorbar(xs, mu, yerr=[mu - lo, hi - mu], color=COL[meth], marker="o", label=LAB[meth],
                        elinewidth=0.6)
        ax.plot([0, 0.11], [0, 0.11], color=COL["ref"], lw=1.0, ls="--", zorder=0, label=r"FPR $=\alpha$")
        ax.set_xlim(0, 0.11)
        ax.set_xticks(alphas)
        ax.set_xlabel(r"Target $\alpha$")
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, ha="left", va="top", fontsize=8)
        ax.set_ylim(0, None)
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.3)
        panel(ax, k, x=-0.24)
    axs[0].set_ylabel("Per-client FPR")
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.17, top=0.82, wspace=0.34)
    save(fig, "F_fpr_vs_alpha")


def f_byzantine():
    C, P, W = csv("T5_calib"), csv("T5_proto"), csv("T5_whitelist")
    fr = [0.1, 0.2, 0.3]
    fig, axs = plt.subplots(1, 4, figsize=(W2, H2))
    eb = {"elinewidth": 0.5, "capsize": 1.5}
    ls_ds = {"NSL": "-", "NBA": "--"}
    mk_ds = {"NSL": "o", "NBA": "s"}
    # (a) calibration poisoning
    ax = axs[0]
    for ds in ("NSL", "NBA"):
        g = C[C.dataset == ds].set_index("frac").loc[fr]
        ax.errorbar(fr, g.asr_global, yerr=g.asr_global_std, color=COL["nodef"], ls=ls_ds[ds], marker=mk_ds[ds], **eb)
        ax.errorbar(fr, g.asr_robust, yerr=g.asr_robust_std, color=COL["robust"], ls=ls_ds[ds], marker=mk_ds[ds], **eb)
    ax.set_xlabel("Calibration poisoning, $f$")
    # (b), (c) prototype poisoning
    for k, ds in ((1, "NSL"), (2, "NBA")):
        ax = axs[k]
        for K, ls in ((1, "-"), (4, "--")):
            for agg, col, mk in (("kmeans", COL["nodef"], "o" if K == 1 else "s"),
                                 ("geomed", COL["robust"], "o" if K == 1 else "s"), ("support", COL["robust"], "D")):
                if K == 1 and agg == "support":
                    continue  # identical to geo-median at K=1 (one prototype kept)
                g = P[(P.dataset == ds) & (P.K == K) & (P.proto_agg == agg)].set_index("frac").loc[fr]
                ax.errorbar(fr, g.asr_global, yerr=g.asr_global_std, color=col, ls=ls if agg != "support" else ":",
                            marker=mk, mfc=col if agg != "support" else "white", **eb)
        ax.set_xlabel(f"Prototype poisoning ({'NSL-KDD' if ds == 'NSL' else 'N-BaIoT'}), $f$")
    # (d) whitelisting, boost 5
    ax = axs[3]
    for ds in ("NSL", "NBA"):
        for agg, col in (("fedavg", COL["nodef"]), ("trimmed", COL["robust"])):
            g = W[(W.dataset == ds) & (W["agg"] == agg) & (W.boost == 5.0) & (W.attack == "whitelist")].set_index(
                "frac").loc[fr]
            ax.errorbar(fr, g.asr_global, yerr=g.asr_global_std, color=col, ls=ls_ds[ds], marker=mk_ds[ds], **eb)
    ax.set_xlabel("Whitelisting (boost 5), $f$")
    for k, ax in enumerate(axs):
        ax.set_xticks(fr)
        ax.set_xlim(0.07, 0.33)
        ax.set_ylim(-5, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        panel(ax, k, x=-0.24)
    axs[0].set_ylabel("Attack success rate (%)")
    handles = [Line2D([], [], color=COL["nodef"], marker="o", label="No defence"),
               Line2D([], [], color=COL["robust"], marker="o",
                      label="Robust (quantile / geo-median / trimmed)"),
               Line2D([], [], color=COL["robust"], ls=":", marker="D", mfc="white", label="Client support ($K{=}4$)"),
               Line2D([], [], color="black", ls="-", marker="o", label="Solid: NSL-KDD (a,d); $K{=}1$ (b,c)"),
               Line2D([], [], color="black", ls="--", marker="s", label="Dashed: N-BaIoT (a,d); $K{=}4$ (b,c)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.0,
               handlelength=2.2)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.17, top=0.8, wspace=0.34)
    save(fig, "F_byzantine")


def f_violation_vs_heterogeneity():
    T = t4()
    orders = {"NSL-KDD": ["iid", "dir1.0", "dir0.5", "dir0.1"], "N-BaIoT": ["dir1.0", "dir0.5", "dir0.1", "natural"]}
    names = {"iid": "IID", "dir1.0": "Dir(1.0)", "dir0.5": "Dir(0.5)", "dir0.1": "Dir(0.1)", "natural": "natural"}
    refs = {}
    for ds, plabel, ph, pre, part in MT.P4SRC:
        refs[(ds, part)] = MT.ref_exact(MT.load_runs(ph, pre, max_seed=5), 0.05)
    fig, axs = plt.subplots(1, 2, figsize=(W1, H1), sharey=True)
    for k, ds in enumerate(("NSL-KDD", "N-BaIoT")):
        ax = axs[k]
        o = orders[ds]
        xs = np.arange(len(o))
        for i, meth in enumerate(METH3):
            g = T[(T.dataset == ds) & (T.method == meth) & (T.alpha == 0.05)].set_index("partition").loc[o]
            ax.errorbar(xs + (i - 1) * 0.08, g.id_frac_violate_mean, yerr=g.id_frac_violate_std, color=COL[meth],
                        marker="o", label=LAB[meth], elinewidth=0.6)
        ax.plot(xs, [refs[(ds, p)] for p in o], color=COL["ref"], ls="--", marker="_", label="Exact-conformal ref.")
        ax.set_xticks(xs)
        ax.set_xticklabels([names[p] for p in o], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.text(0.03, 0.03, ds, transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
        panel(ax, k, x=-0.12 if k else -0.3)
    axs[0].set_ylabel("Share of violating clients")
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.0), columnspacing=1.0)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.25, top=0.78, wspace=0.12)
    save(fig, "F_violation_vs_heterogeneity")


def f_k_tpr1():
    C, F = csv("T3_central_K"), csv("T3_fed_K")
    Ks = [1, 2, 4, 8, 16]
    fig, axs = plt.subplots(1, 2, figsize=(W1, H1), sharey=True)
    # (a) central, lambda = 0
    c = C[(C.dataset == "NSL") & (C.lam == 0.0)].set_index("detector")
    km = c.loc[["template", "kmeans2", "kmeans4", "kmeans8", "kmeans16"]]
    vm = c.loc[["vmf1", "vmf4", "vmf8"]]
    ax = axs[0]
    ax.errorbar(Ks, km["tpr@1"], yerr=km["tpr@1_std"], color=COL["kmeans"], marker="o", label="k-means (max-cos)",
                elinewidth=0.6)
    ax.errorbar(np.array([1, 4, 8]) * 1.08, vm["tpr@1"], yerr=vm["tpr@1_std"], color=COL["vmf"], ls="--", marker="s",
                label="vMF mixture", elinewidth=0.6)
    ax.text(0.97, 0.03, "NSL-KDD central", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    # (b) federated, lambda = 0
    f = F[(F.dataset == "NSL") & (F.lam == 0.0)]
    fk = f[f.sigma == "0.0"].set_index("K").loc[Ks]
    fv = f[f.sigma == "vmf"].set_index("K").loc[Ks]
    ax = axs[1]
    ax.errorbar(Ks, fk["tpr@1"], yerr=fk["tpr@1_std"], color=COL["kmeans"], marker="o", elinewidth=0.6)
    ax.errorbar(np.array(Ks) * 1.08, fv["tpr@1"], yerr=fv["tpr@1_std"], color=COL["vmf"], ls="--", marker="s",
                elinewidth=0.6)
    ax.text(0.97, 0.03, "NSL-KDD federated", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    for k, ax in enumerate(axs):
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ks)
        ax.set_xticklabels([str(k_) for k_ in Ks])
        ax.minorticks_off()
        ax.set_xlabel("Prototypes $K$")
        ax.set_ylim(0, 62)
        panel(ax, k, x=-0.12 if k else -0.3)
    axs[0].set_ylabel("Test TPR at 1% FPR (%)")
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.0))
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.2, top=0.84, wspace=0.12)
    save(fig, "F_K_tpr1")


def f_zeroday_lofo():
    L = csv("T6_lofo")
    g = L[L.detector_mode == "K1"].sort_values("novel_recall@conf_a0.05")
    fig, ax = plt.subplots(figsize=(W1, H1))
    y = np.arange(len(g))
    ax.barh(y, g["novel_recall@conf_a0.05"], xerr=g["novel_recall@conf_a0.05_std"], color=COL["conformal_local"],
            height=0.6, ecolor="black", error_kw={"elinewidth": 0.6, "capsize": 1.5})
    for yy, v, sd in zip(y, g["novel_recall@conf_a0.05"], g["novel_recall@conf_a0.05_std"]):
        ax.text(v + sd + 1.5 if v < 90 else v - 2.5, yy, f"{v:.1f}", va="center",
                ha="left" if v < 90 else "right", fontsize=8, color="black" if v < 90 else "white")
    ax.set_yticks(y)
    ax.set_yticklabels(g.held_out)
    ax.set_xlim(0, 105)
    ax.set_xlabel(r"Recall on the held-out family (%), conformal $\alpha=0.05$")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.2, top=0.97)
    save(fig, "F_zeroday_lofo")


if __name__ == "__main__":
    setup()
    f_teaser()
    f_fpr_vs_alpha()
    f_byzantine()
    f_violation_vs_heterogeneity()
    f_k_tpr1()
    f_zeroday_lofo()
