"""Paper LaTeX tables (tabular only, booktabs) -> paper_assets/tables/*.tex.

Sources (no experiment is re-run, results/ is read-only):
  * raw results/**/metrics.json where the CSV is rounded too coarsely for safe 1-decimal rounding
    (T_repro: CSV has 2 decimals) or where the CSV holds only a Monte-Carlo value (exact-conformal reference);
  * results/tables/*.csv otherwise (3-4 decimals). A guard refuses to round a CSV value that sits on a
    rounding boundary at the target precision (double rounding).
Every raw-recomputed cell is cross-checked against the CSV (same value to the CSV's precision).
Format: mean±std; 1 decimal for % metrics, 3 decimals for |FPR-alpha| and violation shares.
--check: every table is set in a 10-pt twocolumn test document and its natural width is compared with
8.8 cm / 18 cm.

python experiments/make_paper_tables.py [--check]
"""
import argparse
import math
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import betabinom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src.data.paths import PROCESSED  # noqa: E402
from src.utils.select import get, load_runs  # noqa: E402

TAB = "results/tables"
OUT = "paper_assets/tables"
WIDTH = {"single": 8.8, "double": 18.0}  # cm
KIND = {}  # file -> single / double


def _boundary(v, d):
    """True if v (a value read from a rounded CSV) is exactly on a rounding boundary at d decimals."""
    x = abs(v) * 10 ** d
    return abs((x - math.floor(x)) - 0.5) < 1e-7


def num(v, d, src="raw"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if src == "csv" and _boundary(v, d):
        raise ValueError(f"double-rounding risk: CSV value {v} at {d} decimals; recompute from metrics.json")
    s = f"{v:.{d}f}"
    if re.fullmatch(r"-0\.?0*", s):
        s = s[1:]
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def pm(m, s, d, src="raw"):
    return f"{num(m, d, src)}$\\pm${num(s, d, src)}"


def bold(s):
    return f"\\textbf{{{s}}}"


EOR = r" \tabularnewline"  # row end; \\ would be swallowed by \raggedright inside a p-cell


def rrow(cells, rag):
    """One table row. Cells whose index is in `rag` sit in plain p{} columns and get \\raggedright inside the cell,
    so no column spec needs the array package (>{...} without array prints the spec as stray text)."""
    return " & ".join(r"\raggedright " + c if i in rag else c for i, c in enumerate(cells)) + EOR


def best_mask(vals, d, mode):
    """Bold every value whose displayed (rounded) mean equals the best displayed mean (ties at display precision)."""
    shown = [float(f"{v:.{d}f}") if v is not None and not np.isnan(v) else np.nan for v in vals]
    ok = [x for x in shown if not np.isnan(x)]
    if not ok:
        return [False] * len(vals)
    b = min(ok) if mode == "min" else max(ok)
    return [(not np.isnan(x)) and x == b for x in shown]


def write(name, body, kind):
    os.makedirs(OUT, exist_ok=True)
    KIND[name] = kind
    body = body.strip()
    # exactly one tabular per file: no provenance line, no nested tabular, no array-package column syntax
    assert body.startswith(r"\begin{tabular}{") and body.endswith(r"\end{tabular}"), name
    assert body.count(r"\begin{tabular}") == 1 and body.count(r"\end{tabular}") == 1, name
    assert ">{" not in body and r"\arraybackslash" not in body, name
    with open(os.path.join(OUT, name), "w") as f:
        f.write(body + "\n")
    print("wrote", os.path.join(OUT, name))


def csv(name):
    return pd.read_csv(os.path.join(TAB, name + ".csv"))


def check_close(a, b, tol, what):
    if not (abs(a - b) <= tol):
        raise AssertionError(f"{what}: raw {a} vs v2 CSV {b}")


FECO_PAPER = {  # FeCo TDSC 2025 Table IV (central), Table V (novel-attack recall, central), Table VII (FL / self)
    "central": dict(acc=89.55, recall=86.80, fpr=6.82, novel_recall=78.45),
    "fl_iid": dict(acc=85.65, recall=81.15, fpr=8.41),
    "fl_noniid1": dict(acc=86.23, recall=82.10, fpr=8.29),
    "fl_noniid2": dict(acc=85.53, recall=77.90, fpr=4.26),
    "self_noniid2": dict(acc=67.66, recall=46.51, fpr=4.39),
}


def paper1(v):
    """Paper values are exact 2-decimal numbers; round half up to 1 decimal (Decimal, no binary artefacts)."""
    from decimal import ROUND_HALF_UP, Decimal
    return "--" if v is None else str(Decimal(str(v)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def tab_repro():
    rows = [("Central", "central", "finalC_full_s", ("central", "-")),
            ("FL IID", "fl_iid", "finalFL_iid_full_s", ("fl", "iid")),
            ("FL non-IID-1", "fl_noniid1", "finalFL_noniid1_full_s", ("fl", "noniid1")),
            ("FL non-IID-2", "fl_noniid2", "finalFL_noniid2_full_s", ("fl", "noniid2")),
            ("Self-learning non-IID-2", "self_noniid2", "finalSELF_noniid2_full_s", ("self", "noniid2"))]
    T = csv("T_repro")
    lines = []
    for label, key, prefix, (st, pa) in rows:
        runs = load_runs("phase1", prefix, max_seed=5)
        assert len(runs) == 5, (prefix, len(runs))
        cells = [label]
        for met in ("acc", "recall", "fpr", "novel_recall"):
            v = np.array([100 * get(m, "results.template.test.feco_p5")[met] for _, m in runs], float)
            mu, sd = v.mean(), v.std(ddof=1)
            r = T[(T.setting == st) & (T.partition == pa) & (T.metric == met)].iloc[0]
            check_close(mu, r.ours_mean, 0.005 + 1e-9, f"T_repro {label} {met}")
            check_close(sd, r.ours_std, 0.005 + 1e-9, f"T_repro {label} {met} std")
            p = FECO_PAPER[key].get(met)
            if p is not None:
                check_close(p, r.feco_reported, 1e-9, f"paper value {label} {met}")
            cells += [paper1(p), pm(mu, sd, 1)]
        lines.append(" & ".join(cells) + r" \\")
    body = r"""
\begin{tabular}{@{}lcccccccc@{}}
\toprule
 & \multicolumn{2}{c}{Acc (\%)} & \multicolumn{2}{c}{Recall (\%)} & \multicolumn{2}{c}{FPR (\%)} & \multicolumn{2}{c}{Novel recall (\%)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}
Setting & Paper & Ours & Paper & Ours & Paper & Ours & Paper & Ours \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}"""
    write("tab_repro.tex", body, "double")


def _rng(vals, fmt="{:,}"):
    lo, hi = min(vals), max(vals)
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}--{fmt.format(hi)}"


def tab_datasets():
    import json
    info = {}
    for ds, pres in (("NSL", ["phase1/finalC_full_s", "phase6/cen_NSL_feco_s"]), ("NBA", ["phase6/cen_NBA_feco_s"]),
                     ("CIC", ["phase6/cen_CIC_feco_s"])):
        tr, te, dim = [], [], []
        for pre in pres:
            ph, p = pre.split("/")
            for _, m in load_runs(ph, p, max_seed=5):
                t = get(m, "results.template.test.feco_p5")
                tr.append(m["n_train"])
                te.append(t["tp"] + t["tn"] + t["fp"] + t["fn"])
                dim.append(m["dim"])
        info[ds] = (tr, te, dim)
    nb_raw = len(json.load(open(os.path.join(PROCESSED, "nbaiot", "feature_names.json"))))
    cic_raw = len(json.load(open(os.path.join(PROCESSED, "ciciot2023", "audit.json")))["features"])
    # NSL-KDD: official files (125,973 / 22,544 rows; the FeCo protocol trains on the full train file, the corrected protocol on its 80% split)
    nsl_tr, nsl_te, nsl_dim = info["NSL"]
    assert max(nsl_tr) == 125973 and set(nsl_te) == {22544}, (set(nsl_tr), set(nsl_te))
    def recs(tr, te):  # train on line 1, test on line 2 (test may be a seed range)
        t = _rng(te).replace("--", "--\\newline ")
        return f"{_rng(tr)} /\\newline {t}"

    def feat(raw, dims):  # used dimension: max over seeds; the per-seed range goes to the footnote
        return f"{raw}$\\to${max(dims)}"

    rows = [
        ("NSL-KDD", recs([max(nsl_tr)], nsl_te), feat(41, nsl_dim), "Normal + 4 categories", "official files",
         "IID, non-IID-1/2, Dir$^{b}$"),
        ("N-BaIoT", recs(info["NBA"][0], info["NBA"][1]), feat(nb_raw, info["NBA"][2]), "Benign + 10 attacks",
         "time-ordered", "natural, Dir$^{b}$"),
        ("CICIoT\\-2023$^{d}$", recs(info["CIC"][0], info["CIC"][1]), feat(cic_raw, info["CIC"][2]),
         "Benign + 7 families", "file / block$^{c}$", "IID, Dir(0.5)"),
    ]
    rng_txt = "; ".join(f"{n} {min(d)}--{max(d)}" for n, d in (("NSL-KDD", nsl_dim), ("N-BaIoT", info["NBA"][2]),
                                                                ("CICIoT2023", info["CIC"][2])) if min(d) != max(d))
    foot = (f"$^{{a}}$Raw$\\to$used (NSL-KDD after one-hot encoding); the used count varies across seeds ({rng_txt}). "
            r"$^{b}$Dirichlet $\alpha_D\in\{1.0, 0.5, 0.1\}$. "
            r"$^{c}$Split by capture file / contiguous block; the test size depends on the seed. "
            r"$^{d}$Per-capture subset. N-BaIoT: 9 devices, 5 BASHLITE + 5 Mirai attacks; CICIoT2023: 15 attack classes.")
    rag = range(6)
    body = r"""
\begin{tabular}{@{}p{1.2cm}p{1.17cm}p{1.27cm}p{1.17cm}p{1.1cm}p{1.3cm}@{}}
\toprule
""" + rrow(["Dataset", "Train / test records", "Features$^{a}$", "Classes", "Split", "Partitions"], rag) + r"""
\midrule
""" + "\n".join(rrow(r, rag) for r in rows) + r"""
\bottomrule
\multicolumn{6}{@{}p{8.5cm}@{}}{\scriptsize """ + foot + "}" + EOR + r"""
\end{tabular}"""
    write("tab_datasets.tex", body, "single")


P4SRC = [("NSL-KDD", "IID", "phase4", "nsl_iid_s", "iid"), ("NSL-KDD", "Dir(1.0)", "phase4", "nsl_dir1.0_s", "dir1.0"),
         ("NSL-KDD", "Dir(0.5)", "phase3", "fedNSL_lam0.0_s", "dir0.5"),
         ("NSL-KDD", "Dir(0.1)", "phase4", "nsl_dir0.1_s", "dir0.1"),
         ("N-BaIoT", "natural", "phase3", "fedNBA_lam0.0_s", "natural"),
         ("N-BaIoT", "Dir(1.0)", "phase4", "nba_dir1.0_s", "dir1.0"), ("N-BaIoT", "Dir(0.5)", "phase4", "nba_dir0.5_s", "dir0.5"),
         ("N-BaIoT", "Dir(0.1)", "phase4", "nba_dir0.1_s", "dir0.1")]
M3 = ["feco_global", "conformal_global", "conformal_local"]


def ref_exact(runs, a):
    """Expected violation share under EXACT split-conformal thresholds (no Monte Carlo): empirical FPR on m hold-out
    points ~ BetaBinomial(m, k, n+1-k), k = floor(a(n+1)); violation iff FPR > a + 1/(n+1). Mean over clients, then seeds.
    Same client set as fedeval (clients with an ID hold-out)."""
    vals = []
    for _, m in runs:
        r = m["fed_results"]["K1_sigma0.0"]
        o = []
        for n, mm in zip(r["calib_n"], r["id_n"]):
            if mm == 0:
                continue
            k = math.floor(a * (n + 1) + 1e-9)
            if k < 1:
                o.append(0.0)
                continue
            x0 = math.floor((a + 1 / (n + 1)) * mm + 1e-12)
            o.append(float(betabinom.sf(x0, mm, k, n + 1 - k)))
        vals.append(np.mean(o))
    return float(np.mean(vals))


def fpr_cells(a=0.05):
    T = csv("T4_fpr_control")
    out = []
    for ds, plabel, ph, pre, part in P4SRC:
        runs = load_runs(ph, pre, max_seed=5)
        assert len(runs) == 5, (pre, len(runs))
        row = {"ds": ds, "part": plabel, "key": part}
        for meth in M3:
            xs = [m["fed_results"]["K1_sigma0.0"][f"{meth}_a{a}"] for _, m in runs]
            q = {"abs": [x["id"]["mean_abs_fpr_minus_alpha"] for x in xs], "viol": [x["id"]["frac_violate"] for x in xs],
                 "tpr": [100 * x["tpr_mean"] for x in xs], "test": [100 * x["test"]["mean_fpr"] for x in xs]}
            c = T[(T.dataset == ds) & (T.partition == part) & (T.method == meth) & (T.alpha == a)].iloc[0]
            for k, col, sc in (("abs", "id_mean_abs_fpr_minus_alpha", 1), ("viol", "id_frac_violate", 1),
                               ("tpr", "tpr_mean", 100), ("test", "test_mean_fpr", 100)):
                v = np.array(q[k], float)
                row[(meth, k)] = (v.mean(), v.std(ddof=1))
                check_close(v.mean() / sc, c[f"{col}_mean"], 0.00005 + 1e-9, f"T4 {ds} {part} {meth} {k}")
        row["ref"] = ref_exact(runs, a)
        out.append(row)
    return out


def two(a, b):
    """Two-line header cell without makecell or a nested tabular (each file holds exactly one tabular)."""
    return f"\\shortstack{{{a}\\\\{b}}}"


def tab_fpr():
    R = fpr_cells()
    lines, prev = [], None
    for r in R:
        if r["ds"] != prev:
            if prev is not None:
                lines.append(r"\midrule")
            lines.append(f"\\multicolumn{{13}}{{@{{}}l}}{{\\emph{{{r['ds']}}}}} \\\\")
        prev = r["ds"]
        cells = [r["part"]]
        b_abs = best_mask([r[(m, "abs")][0] for m in M3], 3, "min")
        for i, m in enumerate(M3):  # violation share is not bolded: its target is the exact reference, not 0
            a_ = num(r[(m, "abs")][0], 3)
            cells += [bold(a_) if b_abs[i] else a_, num(r[(m, "viol")][0], 3)]
        cells.append(num(r["ref"], 3))
        b_tpr = best_mask([r[(m, "tpr")][0] for m in ("conformal_global", "conformal_local")], 1, "max")
        for i, m in enumerate(("conformal_global", "conformal_local")):
            t_ = num(r[(m, "tpr")][0], 1)
            cells.append(bold(t_) if b_tpr[i] else t_)
        for m in M3:
            cells.append(num(r[(m, "test")][0], 1))
        lines.append(" & ".join(cells) + r" \\")
    A = r"$|\mathrm{FPR}{-}\alpha|$"
    body = (r"""
\begin{tabular}{@{}l""" + "c" * 12 + r"""@{}}
\toprule
 & \multicolumn{2}{c}{FeCo global} & \multicolumn{2}{c}{Global quantile} & \multicolumn{2}{c}{Per-client conformal} & """
            + two("Exact", "ref.") + r""" & \multicolumn{2}{c}{TPR at $\rho$ (\%)} & \multicolumn{3}{c}{Test FPR, shift/drift (\%)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){9-10}\cmidrule(l){11-13}
Partition & """ + f"{A} & Viol. & {A} & Viol. & {A} & Viol. & viol. & Global & Per-cl. & FeCo & Global & Per-cl." + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\multicolumn{13}{@{}p{16cm}@{}}{\scriptsize $\alpha=0.05$; means over 5 seeds. Viol.: share of clients with $\mathrm{FPR}>\alpha+1/(n+1)$ on the exchangeable hold-out; exact ref.: expected share under exact split-conformal thresholds (beta-binomial).} \\
\end{tabular}""")
    write("tab_fpr.tex", body, "double")
    return R


def tab_byz(f=0.3):
    C, P, W = csv("T5_calib"), csv("T5_proto"), csv("T5_whitelist")
    get1 = lambda D, **kw: D[np.logical_and.reduce([D[k] == v for k, v in kw.items()])].iloc[0]  # noqa: E731
    spec = [("Calibration", "Robust quantile",
             lambda ds: (get1(C, dataset=ds, frac=f), "asr_global"), lambda ds: (get1(C, dataset=ds, frac=f), "asr_robust")),
            ("Prototype ($K{=}1$)", "Geo.\\ median",
             lambda ds: (get1(P, dataset=ds, frac=f, K=1, proto_agg="kmeans"), "asr_global"),
             lambda ds: (get1(P, dataset=ds, frac=f, K=1, proto_agg="geomed"), "asr_global")),
            ("Prototype ($K{=}4$)", "Geo.\\ median",
             lambda ds: (get1(P, dataset=ds, frac=f, K=4, proto_agg="kmeans"), "asr_global"),
             lambda ds: (get1(P, dataset=ds, frac=f, K=4, proto_agg="geomed"), "asr_global")),
            ("Prototype ($K{=}4$)", "Client support",
             lambda ds: (get1(P, dataset=ds, frac=f, K=4, proto_agg="kmeans"), "asr_global"),
             lambda ds: (get1(P, dataset=ds, frac=f, K=4, proto_agg="support"), "asr_global")),
            ("Whitelisting (boost 5)", "Trimmed mean",
             lambda ds: (get1(W, dataset=ds, frac=f, boost=5.0, agg="fedavg", attack="whitelist"), "asr_global"),
             lambda ds: (get1(W, dataset=ds, frac=f, boost=5.0, agg="trimmed", attack="whitelist"), "asr_global"))]
    lines = []
    for atk, dfn, none_fn, def_fn in spec:
        cells = [f"{atk}\\newline\\hspace*{{0.6em}}\\emph{{{dfn}}}"]
        for ds in ("NSL", "NBA"):
            (rn, cn), (rd, cd) = none_fn(ds), def_fn(ds)
            vals = [(rn[cn], rn[cn + "_std"]), (rd[cd], rd[cd + "_std"])]
            b = best_mask([v[0] for v in vals], 1, "min")
            for (mu, sd), bb in zip(vals, b):
                s = pm(mu, sd, 1, "csv")
                cells.append(bold(s) if bb else s)
        lines.append(rrow(cells, {0}))
    body = r"""
\begin{tabular}{@{}p{1.95cm}cccc@{}}
\toprule
 & \multicolumn{2}{c}{NSL-KDD ASR (\%)} & \multicolumn{2}{c}{N-BaIoT ASR (\%)}""" + EOR + r"""
\cmidrule(lr){2-3}\cmidrule(l){4-5}
""" + rrow([r"Attack \emph{(defence)}", "None", "Defended", "None", "Defended"], {0}) + r"""
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}"""
    write("tab_byz.tex", body, "single")


def p2(p):
    if p >= 0.995:
        return "1.0"
    return f"{p:.2g}" if p >= 0.01 else f"{p:.2g}"


def tab_h1():
    H = csv("T_H1_valselected")
    lines = []
    for ds, mode, label in (("NSL-KDD", "central", "NSL-KDD central"), ("NSL-KDD", "federated", "NSL-KDD federated"),
                            ("CICIoT2023", "central", "CICIoT central"),
                            ("CICIoT2023", "federated", "CICIoT federated")):
        r = H[(H.dataset == ds) & (H["mode"] == mode)].iloc[0]
        d = num(r.mean_diff_pts, 1, "csv")
        d = ("+" + d) if r.mean_diff_pts > 0 and not d.startswith("-") else d
        ci = f"[{num(r.ci95_lo, 1, 'csv')}, {num(r.ci95_hi, 1, 'csv')}]"
        lines.append(f"{label} & {d} & {ci} & {p2(r.wilcoxon_p)} & {int(r.wins)}/{int(r.ties)}/{int(r.losses)} \\\\")
    pc = H[(H.dataset == "NSL-KDD") & (H["mode"] == "central")].iloc[0].wilcoxon_p
    nset = len(H)
    body = r"""
\begin{tabular}{@{}lcccc@{}}
\toprule
Setting & $\Delta$ (pts) & 95\% CI & \shortstack{$p$\\(Wilcoxon)} & W/T/L \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\multicolumn{5}{@{}l@{}}{\scriptsize Bonferroni-adjusted ($\times$""" + str(nset) + r""") $p$ for NSL-KDD central: """ + \
        f"{min(1.0, nset * pc):.2g}" + r""".} \\
\end{tabular}"""
    write("tab_h1.tex", body, "single")


def tab_methods():
    T = csv("T6_central")
    meths = [("FeCo", "feco"), ("CLAD", "clad"), ("Deep SVDD", "svdd"), ("Deep SAD", "sad"), ("Autoencoder", "ae"),
             ("Isolation Forest", "iforest")]
    cols = [(ds, met) for ds in ("NSL", "NBA", "CIC") for met in ("auroc", "tpr@1")]
    # raw per-seed values (K=1 detector: 'template' for FeCo/CLAD, 'native' otherwise; as tables_phase6.central)
    raw = {}
    for _, m in meths:
        for ds in ("NSL", "NBA", "CIC"):
            runs = load_runs("phase6", f"cen_{ds}_{m}_s", max_seed=5)
            assert len(runs) == 5, (ds, m, len(runs))
            for met in ("auroc", "tpr@1"):
                v = np.array([100 * mm["results"]["template" if "template" in mm["results"] else "native"]["test"]
                              ["ranking"][met] for _, mm in runs], float)
                raw[(ds, m, met)] = (v.mean(), v.std(ddof=1))
                r = T[(T.dataset == ds) & (T.method == m)].iloc[0]
                check_close(v.mean(), r[met], 0.0005 + 1e-9, f"T6_central {ds} {m} {met}")
                check_close(v.std(ddof=1), r[met + "_std"], 0.0005 + 1e-9, f"T6_central {ds} {m} {met} std")
    best = {}
    for ds, met in cols:
        best[(ds, met)] = best_mask([raw[(ds, m, met)][0] for _, m in meths], 1, "max")
    lines = []
    for i, (label, m) in enumerate(meths):
        cells = [label]
        for ds, met in cols:
            s = pm(*raw[(ds, m, met)], 1)
            cells.append(bold(s) if best[(ds, met)][i] else s)
        lines.append(" & ".join(cells) + r" \\")
    body = r"""
\begin{tabular}{@{}lcccccc@{}}
\toprule
 & \multicolumn{2}{c}{NSL-KDD} & \multicolumn{2}{c}{N-BaIoT} & \multicolumn{2}{c}{CICIoT2023} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}
Method & AUROC (\%) & TPR@1\% (\%) & AUROC (\%) & TPR@1\% (\%) & AUROC (\%) & TPR@1\% (\%) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}"""
    write("tab_methods.tex", body, "double")


def _grid(mod, fn, path, keep=lambda c: True):
    import importlib.util
    spec = importlib.util.spec_from_file_location(mod, f"configs/{mod}.py")
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    vals = set()
    for c in getattr(M, fn)():
        if not keep(c):
            continue
        v = get(c, path)
        if v is not None:
            vals.add(v)
    return sorted(vals)


def _set(v):
    return "\\{" + ", ".join(f"{x:g}" for x in v) + "\\}"


def tab_hparams():
    import json
    s1 = json.load(open("configs/phase1_selected.json"))
    s3 = json.load(open("configs/phase3_selected.json"))
    s6 = json.load(open("configs/phase6_selected.json"))
    norm = lambda c: get(c, "train.normalize", True) is not False  # noqa: E731  (raw-dot ablation has its own row)
    g = lambda f, p, mod="phase1_grid": _set(_grid(mod, f, p, norm))  # noqa: E731
    rows = [
        ("Encoder", r"MLP $d$--128--256--128, ReLU", "--", "fixed (FeCo)"),
        ("Optimiser", r"SGD, lr $10^{-3}$", "--", "fixed (FeCo)"),
        ("Embedding norm.", "cosine (normalised)", "normalised / raw dot", "val"),
        (r"$\tau$ (NSL-KDD)", f"{s1['central']['tau']:g}", g("tune_central", "train.tau"), "val"),
        (r"$\tau$ (N-BaIoT / CICIoT)", f"{s3['nba']['tau']:g} / {s6['cic']['tau']:g}",
         g("tune_nba", "train.tau", "phase3_grid"), "val"),
        ("Momentum", f"{s1['central']['momentum']:g}", g("tune_central", "train.momentum"), "val"),
        ("Batch size (NSL central / FL)", f"{s1['central']['batch_size']} / {s1['fl']['batch_size']}",
         g("tune_central", "train.batch_size") + " / " + g("tune_fl", "train.batch_size"), "val"),
        ("Epochs (NSL / NBA, CIC)", f"{s1['central']['epochs']} / {s3['nba']['epochs']}, {s6['cic']['epochs']}",
         g("tune_central", "train.epochs") + " / " + g("tune_nba", "train.epochs", "phase3_grid"), "val"),
        ("FL clients", "50 (NSL, CIC) / 45 (NBA)", "--", "fixed"),
        ("FL rounds / local epochs", "15 / 4", "--", "fixed (FeCo)"),
        (r"FedProx $\mu$", f"{s6['mu_prox']:g}", g("tune_mu", "train.mu_prox", "phase6_grid"), "val"),
        (r"MOON $\mu$", f"{s6['mu_moon']:g}", g("tune_mu", "train.mu_moon", "phase6_grid"), "val"),
        (r"Prototypes $K$", r"per seed (val TPR@1\%)", r"k-means \{1, 2, 4, 8, 16\}; vMF \{4, 8\}", "val"),
        (r"Prototype loss $\lambda$", "0 (headline)", r"\{0, 0.1, 0.5\}", "ablation"),
        (r"Centroid noise $\sigma$", "0 (headline)", r"\{0, 0.01, 0.05, 0.1\}", "ablation"),
        (r"Target $\alpha$", "0.05 (headline)", r"\{0.01, 0.05, 0.1\}", "all reported"),
        ("Calibration size cap", "all", r"\{25, 50, 100, 200, all\}", "ablation"),
        ("FeCo threshold", "5th pct.\\ of train benign", "--", "fixed (FeCo)"),
        ("Malicious fraction $f$", "0.3 (headline)", r"\{0.1, 0.2, 0.3\}", "all reported"),
        ("Boost (proto./calib.\\ / whitelist)", r"3 / \{1, 5\}", "--", "fixed"),
        (r"Trimmed mean $\beta$; robust $f_{\max}$", "0.3; 0.3", "--", "fixed"),
        ("Seeds", r"0--4 (H1: 0--9)", "--", "fixed"),
    ]
    body = r"""
\begin{tabular}{@{}p{2.35cm}p{2.05cm}p{2.1cm}p{1.25cm}@{}}
\toprule
""" + rrow(["Parameter", "Value", "Grid searched", "Selected on"], range(4)) + r"""
\midrule
""" + "\n".join(rrow(r, range(4)) for r in rows) + r"""
\bottomrule
\end{tabular}"""
    write("tab_hparams.tex", body, "single")


TEST_DOC = r"""\documentclass[10pt,twocolumn]{article}
%% No array package on purpose: the tables must compile with only booktabs (-halt-on-error catches any column spec
%% that needs array, e.g. >{...}).
\usepackage[paperwidth=8.5in,paperheight=11in,textwidth=7.16in,columnsep=0.17in]{geometry}
%% Font: Computer Modern. Times metrics (ptm / newtx / IEEEtran.cls) are not installed in this environment and no
%% system package may be added; CM glyphs are at least as wide as Times for every glyph used (digits equal, letters
%% and bold wider), so a table that fits here fits in IEEEtran (Times).
%% booktabs / multirow are not installed either: width-neutral
%% stand-ins. Rules do not change a tabular's width; \cmidrule(trim){a-b} is rewritten to \cline{a-b} in the test copy.
\providecommand\toprule{\hline}\providecommand\midrule{\hline}\providecommand\bottomrule{\hline}
\newsavebox\tb
\begin{document}
\footnotesize\setlength\tabcolsep{4pt}
%s
\end{document}
"""


def _compile_one(tmp, name):
    txt = open(os.path.join(OUT, name)).read()
    txt = re.sub(r"\\cmidrule(\([a-z]*\))?\{([0-9]+-[0-9]+)\}", r"\\cline{\2}", txt)
    src = os.path.join(tmp, name)
    open(src, "w").write(txt)
    doc = TEST_DOC % f"\\sbox\\tb{{\\input{{{src}}}}}\\typeout{{TABWIDTH \\the\\wd\\tb}}"
    stem = name[:-4] + "_test"
    open(os.path.join(tmp, stem + ".tex"), "w").write(doc)
    p = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem + ".tex"], cwd=tmp,
                       capture_output=True, text=True)
    log = open(os.path.join(tmp, stem + ".log"), errors="replace").read()
    if p.returncode != 0:
        print(log[-3000:])
        raise SystemExit(f"test compile failed: {name}")
    w = float(re.search(r"TABWIDTH ([\d.]+)pt", log).group(1))
    over = [float(x) for x in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)]
    return w, over


def check():
    os.makedirs(OUT, exist_ok=True)
    rep, bad = [], False
    with tempfile.TemporaryDirectory() as tmp:
        for name in sorted(KIND):
            w, over = _compile_one(tmp, name)
            lim = WIDTH[KIND[name]] / 2.54 * 72.27
            ok = w <= lim and not over
            bad |= not ok
            rep.append(f"{name:20s} {KIND[name]:6s} width {w / 72.27 * 2.54:5.2f} cm  limit {WIDTH[KIND[name]]:4.1f} cm  "
                       f"overfull cells {len(over)} (max {max(over, default=0):.1f}pt)  {'OK' if ok else 'FAIL'}")
    txt = "\n".join(rep)
    print(txt)
    open(os.path.join(OUT, "width_check.txt"), "w").write(
        "Natural width of each tabular at \\footnotesize, \\tabcolsep=4pt, 10pt twocolumn test doc, Computer Modern\n"
        "(Times not installed; CM is at least as wide as Times for the glyphs used -> conservative).\n"
        "booktabs rules replaced by width-neutral stand-ins (package not installed here). 'overfull cells' = text wider\n"
        "than its p-column (Overfull \\hbox inside the table).\n" + txt + "\n")
    if bad:
        raise SystemExit("table(s) too wide")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    tab_repro()
    tab_datasets()
    tab_fpr()
    tab_byz()
    tab_h1()
    tab_methods()
    tab_hparams()
    if a.check:
        check()
