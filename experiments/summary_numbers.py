"""Headline numbers recomputed from raw metrics.json (never from rounded CSVs) for the CPU archive (results/_v1)
and the current results (results). Writes results/tables/T8_changes_v1_v2.csv with old → new, Δ, the
devices/seeds behind each number and the reason it may have moved.
python experiments/summary_numbers.py"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def runs(root, phase, prefix, max_seed=5, exact_suffix=True):
    out = []
    for d in sorted(glob.glob(os.path.join(root, phase, prefix + "*"))):
        rid = os.path.basename(d)
        if exact_suffix and not re.fullmatch(re.escape(prefix) + r"\d+", rid):
            continue
        mp = os.path.join(d, "metrics.json")
        if os.path.exists(mp):
            c = yaml.safe_load(open(os.path.join(d, "config.yaml")))
            if c.get("seed", 0) < max_seed:
                out.append((c, json.load(open(mp))))
    return out


def ms(vals):
    v = np.array([x for x in vals if x is not None], float)
    return (float(v.mean()) if len(v) else np.nan, float(v.std(ddof=1)) if len(v) > 1 else np.nan, len(v))


def dev(R):
    return "+".join(sorted({m.get("device", "cpu") for _, m in R}))


def get(d, path):
    for k in path.split("/"):
        d = d[k]
    return d


def thr_name(m, name):
    """Earlier FeCo-protocol runs (train_on='full') were migrated conf_a* → invalid_conf_a*; FeCo p5 is unaffected."""
    return name


class Collector:
    def __init__(self, root):
        self.root, self.rows = root, []

    def add(self, claim, R, vals, scale=100.0):
        mu, sd, n = ms(vals)
        self.rows.append({"claim": claim, "mean": mu * scale if mu == mu else mu, "std": sd * scale if sd == sd else sd,
                          "n": n, "device": dev(R) if R else ""})

    def scalar(self, claim, value, R=None, n=None):
        self.rows.append({"claim": claim, "mean": value, "std": np.nan, "n": n if n is not None else (len(R) if R else 0),
                          "device": dev(R) if R else ""})


def collect(root):
    C = Collector(root)
    r = lambda ph, pre, ms_=5: runs(root, ph, pre, ms_)  # noqa: E731
    R = r("phase1", "finalC_full_s")
    for k in ("acc", "recall", "precision", "f1", "fpr"):
        C.add(f"P1 central FeCo-protocol {k}", R, [get(m, f"results/template/test/feco_p5/{k}") for _, m in R])
    C.add("P1 central novel-attack recall (p5)", R, [m["results"]["template"]["test"]["feco_p5"].get("novel_recall") for _, m in R])
    for cat in ("DoS", "Probe", "R2L", "U2R"):
        C.add(f"P1 central recall {cat} (p5)", R,
              [m["results"]["template"]["test"]["feco_p5"].get("recall_by_group", {}).get(cat) for _, m in R])
    for sch in ("iid", "noniid1", "noniid2"):
        R = r("phase1", f"finalFL_{sch}_full_s")
        C.add(f"P1 FL {sch} acc", R, [get(m, "results/template/test/feco_p5/acc") for _, m in R])
        C.add(f"P1 FL {sch} novel recall", R, [m["results"]["template"]["test"]["feco_p5"].get("novel_recall") for _, m in R])
    R = r("phase1", "finalSELF_noniid2_full_s")
    C.add("P1 self-learning noniid2 acc", R, [get(m, "results/template/test/feco_p5/acc") for _, m in R])
    R = r("phase1", "debugC_A3_sel_log1p_s")
    for k in ("acc", "recall", "fpr"):
        C.add(f"P1 debug A3 log1p {k}", R, [get(m, f"results/template/test/feco_p5/{k}") for _, m in R])
    R = r("phase2", "corrC_s")
    for k in ("acc", "fpr"):
        C.add(f"P2 corrected central conf0.05 {k}", R, [get(m, f"results/template/test/conf_a0.05/{k}") for _, m in R])
    C.add("P2 corrected central AUROC", R, [get(m, "results/template/test/ranking/auroc") for _, m in R])
    C.add("P2 corrected central novel recall conf0.05", R,
          [m["results"]["template"]["test"]["conf_a0.05"].get("novel_recall") for _, m in R])
    Rs = r("phase2", "splitpaperC_s")
    C.add("P2 80% split paper-FS central acc p5", Rs, [get(m, "results/template/test/feco_p5/acc") for _, m in Rs])
    C.add("P2 corrected (FS on split) central acc p5", R, [get(m, "results/template/test/feco_p5/acc") for _, m in R])
    same = [sorted(a[1]["dropped_features"]) == sorted(b[1]["dropped_features"]) for a, b in zip(R, Rs)]
    C.scalar("P2 central FS-on-split == paper drop list (share of seeds)", float(np.mean(same)) if same else np.nan, R)
    for sch in ("iid", "noniid1", "noniid2"):
        Rf, Rp = r("phase2", f"corrFL_{sch}_s"), r("phase2", f"splitpaperFL_{sch}_s")
        C.add(f"P2 FL {sch} federated FS acc p5", Rf, [get(m, "results/template/test/feco_p5/acc") for _, m in Rf])
        C.add(f"P2 FL {sch} pooled paper FS acc p5", Rp, [get(m, "results/template/test/feco_p5/acc") for _, m in Rp])
    for name, opts, fmt in (("ANOVA-K", range(5), "leakK{}_s{}"), ("tau", [0.05, 0.1, 0.2, 0.5], "leakTau{}_s{}")):
        opt, devs = [], []
        for s in range(5):
            res = []
            for o in opts:
                x = runs(root, "phase2", fmt.format(o, s), 5, exact_suffix=False)
                x = [y for y in x if y[0]["run_id"] == fmt.format(o, s)]
                if x:
                    t = x[0][1]["results"]["template"]
                    res.append((t["val"]["ranking"]["tpr@1"], t["val"]["ranking"]["auroc"], t["test"]["feco_p5"]["acc"]))
                    devs.append(x[0])
            if len(res) == len(list(opts)):
                v = max(res, key=lambda z: (round(z[0], 4), z[1]))
                t = max(res, key=lambda z: z[2])
                opt.append(t[2] - v[2])
        C.add(f"P2 test-selection optimism {name} (acc pts)", devs, opt)
    for cat in ("DoS", "Probe", "R2L", "U2R"):
        R = r("phase2", f"loco_{cat}_s")
        C.add(f"P2 LOCO zero-day recall {cat} conf0.05", R, [get(m, "results/template/zeroday/conf_a0.05/novel_recall") for _, m in R])
    R = r("phase3", "cenNSL_lam0.0_s")
    for d in ("template", "kmeans4", "kmeans16", "vmf4", "vmf8"):
        C.add(f"P3 NSL central TPR@1 {d}", R, [m["results"][d]["test"]["ranking"]["tpr@1"] for _, m in R])
    R = r("phase3", "cenNSL_lam0.5_s")
    C.add("P3 NSL central TPR@1 vmf8 lam0.5", R, [m["results"]["vmf8"]["test"]["ranking"]["tpr@1"] for _, m in R])
    R0, R1 = r("phase3", "fedNSL_lam0.0_s"), r("phase3", "fedNSL_lam0.1_s")
    C.add("P3 NSL fed TPR@1 K1 lam0", R0, [m["fed_results"]["K1_sigma0.0"]["ranking_test"]["tpr@1"] for _, m in R0])
    C.add("P3 NSL fed TPR@1 K4 lam0.1", R1, [m["fed_results"]["K4_sigma0.0"]["ranking_test"]["tpr@1"] for _, m in R1])
    for K, sg in ((1, 0.0), (1, 0.05), (16, 0.0), (16, 0.05)):
        C.add(f"P3 NSL fed TPR@1 K{K} sigma{sg}", R0, [m["fed_results"][f"K{K}_sigma{sg}"]["ranking_test"]["tpr@1"] for _, m in R0])
    srcs = {"iid": ("phase4", "nsl_iid_s"), "dir0.1": ("phase4", "nsl_dir0.1_s"), "dir0.5": ("phase3", "fedNSL_lam0.0_s"),
            "dir1.0": ("phase4", "nsl_dir1.0_s")}
    test_fpr = {}
    for part, (ph, pre) in srcs.items():
        R = r(ph, pre)
        for meth in ("conformal_local", "conformal_global", "feco_global", "robust_global"):
            key = f"{meth}_a0.05"
            vals = [m["fed_results"]["K1_sigma0.0"][key] for _, m in R]
            if part in ("iid", "dir0.1"):
                C.add(f"P4 NSL {part} |FPR-a| id {meth}", R, [v["id"]["mean_abs_fpr_minus_alpha"] for v in vals], 1.0)
            test_fpr[(part, meth)] = ms([v["test"]["mean_fpr"] for v in vals])[0]
            if part == "dir0.1" and meth in ("conformal_local", "conformal_global"):
                C.add(f"P4 NSL dir0.1 mean TPR {meth}", R, [v["tpr_mean"] for v in vals], 1.0)
    for meth in ("conformal_local", "conformal_global", "feco_global", "robust_global"):
        v = [x for (p, mth), x in test_fpr.items() if mth == meth]
        if v:
            C.scalar(f"P4 KDDTest+ mean test FPR a0.05 {meth} (min over 4 partitions)", min(v), n=len(v))
            C.scalar(f"P4 KDDTest+ mean test FPR a0.05 {meth} (max over 4 partitions)", max(v), n=len(v))
    # N-BaIoT multi-prototype / conformal cells were not re-run on GPU → the same CPU runs in both roots
    nba = {"natural": ("phase3", "fedNBA_lam0.0_s"), "dir0.1": ("phase4", "nba_dir0.1_s"),
           "dir0.5": ("phase4", "nba_dir0.5_s"), "dir1.0": ("phase4", "nba_dir1.0_s")}
    for part, (ph, pre) in nba.items():
        R = r(ph, pre)
        for meth in ("conformal_local", "conformal_global", "feco_global"):
            vals = [m["fed_results"]["K1_sigma0.0"][f"{meth}_a0.05"]["id"] for _, m in R]
            C.add(f"P4 NBA {part} id mean FPR {meth} a0.05", R, [v["mean_fpr"] for v in vals], 1.0)
            C.add(f"P4 NBA {part} id violation share {meth} a0.05", R, [v["frac_violate"] for v in vals], 1.0)
            C.add(f"P4 NBA {part} id |FPR-a| {meth} a0.05", R, [v["mean_abs_fpr_minus_alpha"] for v in vals], 1.0)
    for ds in ("NSL", "NBA"):
        R = r("phase5", f"{ds}_clean_fedavg_s")
        if not R:
            continue
        ph = lambda k: [m["byz_posthoc"][k]["byz"] for _, m in R if k in m.get("byz_posthoc", {})]  # noqa: E731
        for f in (0.1, 0.2, 0.3):
            ck = f"attackcalib_frac{f}_boost3.0_f_max0.3"
            C.add(f"P5 {ds} calib f{f} ASR global", R, [b["asr_global"] for b in ph(ck)])
            C.add(f"P5 {ds} calib f{f} ASR robust", R, [b["asr_robust"] for b in ph(ck)])
            for K in (1, 4):
                for pa in ("kmeans", "geomed", "trimmed", "support"):
                    k = f"attackproto_frac{f}_boost3.0_K{K}_proto_agg{pa}"
                    C.add(f"P5 {ds} proto K{K} f{f} {pa} ASR", R, [b["asr_global"] for b in ph(k)])
        for f in (0.2, 0.3):
            Rw = r("phase5", f"{ds}_wl_f{f}_b5.0_fedavg_s")
            C.add(f"P5 {ds} whitelist f{f} b5 fedavg ASR", Rw,
                  [m["fed_results"]["K1_sigma0.0"]["byz"].get("asr_global") for _, m in Rw])
            for agg in ("trimmed", "median"):
                Rw = r("phase5", f"{ds}_wl_f{f}_b5.0_{agg}_s")
                C.add(f"P5 {ds} whitelist f{f} b5 {agg} ASR", Rw,
                      [m["fed_results"]["K1_sigma0.0"]["byz"].get("asr_global") for _, m in Rw])
    for ds in ("NSL", "NBA", "CIC"):
        for meth in ("feco", "clad", "ae", "sad", "svdd", "iforest"):
            R = r("phase6", f"cen_{ds}_{meth}_s")
            if not R:
                continue
            det = "template" if meth in ("feco", "clad") else "native"
            for k in ("auroc", "tpr@1"):
                C.add(f"P6 central {ds} {meth} {k}", R, [m["results"][det]["test"]["ranking"][k] for _, m in R])
    for fam in ("DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"):
        R = r("phase6", f"lofo_CIC_{fam}_s")
        C.add(f"P6 LOFO CIC {fam} zero-day recall conf0.05", R,
              [m["results"]["template"]["test"]["conf_a0.05"].get("novel_recall") for _, m in R])
    # FedProx/MOON vs FedAvg (K=1 template / native), max |Δ| over cells and metrics
    cells = {}
    for c, m in runs(root, "phase6", "fl_", 5, exact_suffix=False):
        _, ds, part, meth, algo, _s = c["run_id"].split("_")
        for det, res in m["results"].items():
            fam = "K1" if det in ("template", "native") else "MP"
            for k in ("auroc", "auprc", "tpr@1", "tpr@5"):
                cells.setdefault((ds, part, meth, det, fam, k), {}).setdefault(algo, []).append(res["test"]["ranking"][k])
    diffs = []
    for key, d in cells.items():
        if "fedavg" in d:
            for algo in ("fedprox", "moon"):
                if algo in d:
                    diffs.append((abs(np.mean(d[algo]) - np.mean(d["fedavg"])) * 100, key, algo))
    FL_DIFFS[root] = diffs
    # the T6_fl table cells (128): K=1 template and the per-seed val-selected multi-prototype detector
    tcell = {}
    for c, m in runs(root, "phase6", "fl_", 5, exact_suffix=False):
        _, ds, part, meth, algo, _s = c["run_id"].split("_")
        if meth not in ("feco", "clad"):
            continue
        res = m["results"]
        cands = [k for k in res if k.startswith(("template", "kmeans", "vmf"))]
        sel = max(cands, key=lambda d: (round(res[d]["val"]["ranking"]["tpr@1"], 4), res[d]["val"]["ranking"]["auroc"]))
        for fam, det in (("K1", "template"), ("MP", sel)):
            for k in ("auroc", "auprc", "tpr@1", "tpr@5"):
                tcell.setdefault((ds, part, meth, fam, k), {}).setdefault(algo, []).append(res[det]["test"]["ranking"][k])
    td = [(abs(np.mean(d[a]) - np.mean(d["fedavg"])) * 100, key, a) for key, d in tcell.items() if "fedavg" in d
          for a in ("fedprox", "moon") if a in d]
    for fam_name, sel_ in (("T6_fl cells, K=1", lambda k: k[3] == "K1"), ("T6_fl cells, val-selected MP", lambda k: k[3] == "MP"),
                           ("T6_fl cells, all", lambda k: True)):
        dd = [x for x in td if sel_(x[1])]
        if dd:
            mx = max(dd, key=lambda z: z[0])
            C.scalar(f"P6 FedProx/MOON vs FedAvg max |Δ| pts ({fam_name})", mx[0], n=len(dd))
            C.rows[-1]["note"] = f"worst: {'/'.join(mx[1][:4])} {mx[1][4]} {mx[2]}"
            C.scalar(f"P6 FedProx/MOON vs FedAvg cells > 0.3 pts ({fam_name})", float(sum(x[0] > 0.3 for x in dd)), n=len(dd))
    for fam_name, sel in (("K=1 detectors", lambda k: k[4] == "K1"), ("FeCo K=1 only", lambda k: k[4] == "K1" and k[2] == "feco"),
                          ("multi-prototype detectors", lambda k: k[4] == "MP"), ("all detectors", lambda k: True)):
        dd = [x for x in diffs if sel(x[1])]
        if dd:
            mx = max(dd, key=lambda z: z[0])
            C.scalar(f"P6 FedProx/MOON vs FedAvg max |Δ| pts ({fam_name})", mx[0], n=len(dd))
            C.scalar(f"P6 FedProx/MOON vs FedAvg cells > 0.3 pts ({fam_name})", float(sum(x[0] > 0.3 for x in dd)), n=len(dd))
            C.rows[-2]["note"] = f"worst: {'/'.join(mx[1][:4])} {mx[1][5]} {mx[2]}"
    return pd.DataFrame(C.rows)


FL_DIFFS = {}


REASON_RULES = [("P1 ", "test-file fix (D-025) + GPU"), ("P2 LOCO", "GPU (zero-day split is from KDDTrain+, test file irrelevant)"),
                ("P2 ", "test-file fix (D-025) + GPU"), ("P3 ", "test-file fix (D-025) + GPU + B4"),
                ("P4 ", "test-file fix (D-025) + GPU"), ("P5 NSL", "test-file fix + GPU + 5 seeds (v1: 3)"),
                ("P5 NBA", "GPU + 5 seeds (v1: 3)"), ("P6 central CIC", "CICIoT split v2 (D-026) + GPU + 5 seeds"),
                ("P6 LOFO", "CICIoT split v2 (D-026) + GPU + 5 seeds"), ("P6 central NSL", "test-file fix + GPU + 5 seeds"),
                ("P6 central NBA", "GPU + 5 seeds (v1: 3)"), ("P6 FedProx", "test-file fix / CIC split v2 + GPU + seeds")]


def reason(claim):
    for pre, why in REASON_RULES:
        if claim.startswith(pre):
            return why
    return ""


def main():
    old, new = collect("results/_v1"), collect("results")
    df = old.merge(new, on="claim", how="outer", suffixes=("_v1", "_v2"))
    df["delta"] = df["mean_v2"] - df["mean_v1"]
    df["reason"] = df.claim.map(reason)
    order = {c: i for i, c in enumerate(list(new.claim) + [c for c in old.claim if c not in set(new.claim)])}
    df = df.sort_values("claim", key=lambda s: s.map(order))
    df.to_csv("results/tables/T8_changes_v1_v2.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(df[["claim", "mean_v1", "std_v1", "n_v1", "mean_v2", "std_v2", "n_v2", "device_v2", "delta"]].to_string())


if __name__ == "__main__":
    main()
