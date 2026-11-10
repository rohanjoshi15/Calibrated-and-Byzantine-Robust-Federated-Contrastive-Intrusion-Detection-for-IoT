"""Leakage-free protocol on NSL-KDD. STAGE: corrected | leakage | loco.
corrected: 80% train split, feature selection on the train split (central: pooled procedure; FL: federated FS),
val-tuned hyper-parameters, thresholds FeCo p5 (train benign) and split-conformal α on held-out calib; 5 seeds
(split_seed=seed).
leakage: optimism of FeCo's protocol: (a) ANOVA K∈{0..4}, val-choice vs test-choice; (b) τ, val vs test.
loco: leave-one-category-out zero-day recall."""
import itertools
import json
import os

S1 = json.load(open("configs/phase1_selected.json"))
FL = {"n_clients": 50, "rounds": 15, "local_epochs": 4}


def corrected():
    out = []
    for s in range(5):
        out.append({"phase": "phase2", "run_id": f"corrC_s{s}", "seed": s, "split_seed": s,
                    "data": {"name": "nslkdd", "fs": "procedure"}, "mode": "central", "train": S1["central"],
                    "eval": {"test": True}})
        for sch in ("iid", "noniid1", "noniid2"):
            out.append({"phase": "phase2", "run_id": f"corrFL_{sch}_s{s}", "seed": s, "split_seed": s,
                        "data": {"name": "nslkdd", "fs": "federated"}, "mode": "fl", "train": S1["fl"],
                        "fl": {**FL, "scheme": sch}, "eval": {"test": True}})
            # FeCo-protocol FL on the same split for a paired comparison (paper FS, same threshold rules)
            out.append({"phase": "phase2", "run_id": f"splitpaperFL_{sch}_s{s}", "seed": s, "split_seed": s,
                        "data": {"name": "nslkdd", "fs": "paper"}, "mode": "fl", "train": S1["fl"],
                        "fl": {**FL, "scheme": sch}, "eval": {"test": True}})
        out.append({"phase": "phase2", "run_id": f"splitpaperC_s{s}", "seed": s, "split_seed": s,
                    "data": {"name": "nslkdd", "fs": "paper"}, "mode": "central", "train": S1["central"],
                    "eval": {"test": True}})
    return out


def leakage():
    out = []
    for K, s in itertools.product(range(5), range(5)):
        out.append({"phase": "phase2", "run_id": f"leakK{K}_s{s}", "seed": s, "split_seed": s,
                    "data": {"name": "nslkdd", "fs": "procedure", "n_anova": K}, "mode": "central",
                    "train": S1["central"], "eval": {"test": True}})
    for tau, s in itertools.product([0.05, 0.1, 0.2, 0.5], range(5)):
        out.append({"phase": "phase2", "run_id": f"leakTau{tau}_s{s}", "seed": s, "split_seed": s,
                    "data": {"name": "nslkdd", "fs": "procedure"}, "mode": "central",
                    "train": {**S1["central"], "tau": tau}, "eval": {"test": True}})
    return out


def loco():
    out = []
    for c, s in itertools.product(["DoS", "Probe", "R2L", "U2R"], range(5)):
        out.append({"phase": "phase2", "run_id": f"loco_{c}_s{s}", "seed": s, "split_seed": s,
                    "data": {"name": "nslkdd", "fs": "procedure", "holdout": c}, "mode": "central",
                    "train": S1["central"], "eval": {"test": True}})
    return out


def configs():
    st = os.environ.get("STAGE", "corrected")
    return sum((globals()[x]() for x in st.split(",")), [])
