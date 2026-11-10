"""CPU vs GPU equivalence. For each pair (CPU metrics.json in results/_v1, cuda re-run of the SAME config), flatten
all numeric leaves and report the max absolute difference per metric name (last key of the path).
Pairs: the 3 designated equivalence runs (results/phase8_equiv, kmeans_impl=legacy so only the device differs) and,
as a larger sample, every identical-config N-BaIoT Byzantine / full-evaluation re-run (seeds 0-2).
Also checks the GPU determinism pair (results/phase8_check detA/detB: must be bit-identical).
Outputs: results/tables/T8_cpu_vs_gpu.csv (designated), T8_cpu_vs_gpu_nbaiot_reruns.csv, T8_gpu_determinism.csv.
python experiments/phase8_equiv.py"""
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml

DESIGNATED = [("P2 (NSL-KDD, v1 mirror test file)", "phase8_equiv/eqP2_leakK4_s1_mirror", "phase2/leakK4_s1"),
              ("P3 (N-BaIoT central, substitutes P2)", "phase8_equiv/eqP3_cenNBA_lam0.0_s0", "phase3/cenNBA_lam0.0_s0"),
              ("P4 (N-BaIoT federated Dir1.0)", "phase8_equiv/eqP4_nba_dir1.0_s4", "phase4/nba_dir1.0_s4"),
              ("P5 (N-BaIoT whitelisting f0.2 b1)", "phase5/NBA_wl_f0.2_b1.0_fedavg_s1", "phase5/NBA_wl_f0.2_b1.0_fedavg_s1")]
SKIP = {"runtime_s", "loss", "steps"}
METRICS = {"auroc", "auprc", "tpr@1", "tpr@5", "acc", "recall", "precision", "f1", "fpr", "rho", "mean_fpr",
           "mean_abs_fpr_minus_alpha", "frac_violate", "tpr_mean", "asr_global", "asr_robust", "novel_recall"}


def flat(o, pre=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, pre + (str(k),))
    elif isinstance(o, list):
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in o) and len(o) < 2000:
            for i, x in enumerate(o):
                yield pre + (f"[{i}]",), x
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield pre, float(o)


HEADLINE = ("results/template/test/ranking/", "results/template/test/conf_a0.05/", "results/template/test/feco_p5/",
            "fed_results/K1_sigma0.0/ranking_test/", "fed_results/K1_sigma0.0/conformal_local_a0.05/",
            "fed_results/K1_sigma0.0/byz/", "results/native/test/ranking/", "results/template/zeroday/")


def compare(a, b, only=None):
    """only: tuple of path prefixes ('a/b/') to restrict to (e.g. HEADLINE); None = all numeric leaves."""
    A, B = dict(flat(a)), dict(flat(b))
    rows = {}
    for k in set(A) & set(B):
        if only is not None and not "/".join(k).startswith(only):
            continue
        leaf = next((x for x in reversed(k) if not x.startswith("[")), k[-1])
        if leaf in SKIP or "history" in k or "train_info" in k or "partition" in k:
            continue
        x, y = A[k], B[k]
        if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
            continue
        rows.setdefault(leaf, []).append(abs(x - y))
    return rows


def main():
    out = []
    for name, new, old in DESIGNATED:
        pn, po = os.path.join("results", new, "metrics.json"), os.path.join("results", "_v1", old, "metrics.json")
        if not (os.path.exists(pn) and os.path.exists(po)):
            out.append({"pair": name, "note": "missing"})
            continue
        mn, mo = json.load(open(pn)), json.load(open(po))
        for scope, only in (("all evaluations", None), ("headline (K=1, α=0.05)", HEADLINE)):
            rows = compare(mo, mn, only)
            for leaf in sorted(rows):
                if leaf == "rho" and scope == "all evaluations":
                    continue  # vMF log-likelihood thresholds are unbounded; differences not comparable across metrics
                if leaf in METRICS or leaf.startswith("tpr@") or leaf.startswith("asr"):
                    v = np.array(rows[leaf])
                    out.append({"pair": name, "scope": scope, "new": new, "v1": old, "device_new": mn.get("device"),
                                "metric": leaf, "n_values": len(v), "max_abs_diff": v.max(), "mean_abs_diff": v.mean(),
                                "frac_identical": float(np.mean(v == 0))})
    D = pd.DataFrame(out)
    D.to_csv("results/tables/T8_cpu_vs_gpu.csv", index=False, float_format="%.6g")
    # larger sample: N-BaIoT identical-config re-runs (seeds 0-2)
    big = []
    for ph, pat in (("phase5", "NBA_*_s[012]"), ("phase6", "cen_NBA_*_s[012]"), ("phase6", "fl_NBA_*_s[012]"),
                    ("phase6", "dev_*_s[012]")):
        for d in sorted(glob.glob(os.path.join("results", ph, pat))):
            rid = os.path.basename(d)
            pn, po = os.path.join(d, "metrics.json"), os.path.join("results", "_v1", ph, rid, "metrics.json")
            if not (os.path.exists(pn) and os.path.exists(po)):
                continue
            mn = json.load(open(pn))
            if mn.get("device") != "cuda":
                continue
            cn, co = yaml.safe_load(open(os.path.join(d, "config.yaml"))), \
                yaml.safe_load(open(os.path.join("results", "_v1", ph, rid, "config.yaml")))
            cn = {k: v for k, v in cn.items() if k != "device"}
            if cn != co:
                continue
            for leaf, v in compare(json.load(open(po)), mn, HEADLINE).items():
                if leaf in ("auroc", "auprc", "tpr@1", "tpr@5", "acc", "recall", "fpr", "tpr_mean", "asr_global", "mean_fpr"):
                    big.append({"phase": ph, "run_id": rid, "metric": leaf, "max_abs_diff": max(v)})
    B = pd.DataFrame(big)
    if len(B):
        S = B.groupby("metric").max_abs_diff.agg(["count", "median", "max"]).reset_index()
        S["n_runs"] = B.run_id.nunique()
        S["devices"] = "cpu (v1) vs cuda (Phase 8)"
        S.to_csv("results/tables/T8_cpu_vs_gpu_nbaiot_reruns.csv", index=False, float_format="%.6g")
        print(S.to_string())
    # determinism pair
    a, b = "results/phase8_check/detA_corrC_s0/metrics.json", "results/phase8_check/detB_corrC_s0/metrics.json"
    if os.path.exists(a) and os.path.exists(b):
        rows = compare(json.load(open(a)), json.load(open(b)))
        n = sum(len(v) for v in rows.values())
        mx = max(max(v) for v in rows.values())
        pd.DataFrame([{"pair": "detA vs detB (corrC_s0, cuda)", "n_numeric_fields": n, "max_abs_diff": mx,
                       "bit_identical": mx == 0, "device": "cuda"}]).to_csv("results/tables/T8_gpu_determinism.csv", index=False)
        print("determinism:", n, "fields, max abs diff", mx)
    if len(D) and "metric" in D:
        pd.set_option("display.width", 250)
        print(D.groupby(["scope", "pair", "metric"]).max_abs_diff.max().unstack(1).to_string())


if __name__ == "__main__":
    main()
