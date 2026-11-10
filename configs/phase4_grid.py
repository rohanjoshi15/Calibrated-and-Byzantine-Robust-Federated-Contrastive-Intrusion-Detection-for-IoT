"""Conformal thresholds (C2). Per-client evaluation via mode 'fed'.
NSL-KDD: 50 clients; benign heterogeneity over 10 service groups and attack categories, both IID or Dirichlet(α_D),
α_D ∈ {0.1, 0.5, 1.0} (same α_D for benign groups and attack categories).
N-BaIoT: 45 clients; 'natural' (9 devices × 5 gateways) and Dirichlet(α_D) over devices.
Thresholds (a) FeCo global/local p=100α, (b) per-client conformal (floor & ceil), (c) federated histogram global
quantile, (d) robust quantile; α ∈ {0.01, 0.05, 0.1}; calibration-size caps {25, 50, 100, 200} (n ablation)."""
import itertools
import json
import os

S1 = json.load(open("configs/phase1_selected.json"))
S3 = "configs/phase3_selected.json"
DET = {"K_fed": [1, 4], "calib_caps": [25, 50, 100, 200]}


def configs():
    out = []
    parts = [("iid", None)] + [("dirichlet", a) for a in (0.1, 1.0)]  # Dir(0.5) = phase3 fedNSL_lam0.0 (reused)
    for (sch, a), s in itertools.product(parts, range(5)):
        tag = sch if a is None else f"dir{a}"
        fl = {"n_clients": 50, "rounds": 15, "local_epochs": 4, "benign_scheme": sch,
              "attack_scheme": "iid" if a is None else "dirichlet"}
        if a is not None:
            fl.update(benign_alpha=a, attack_alpha=a)
        out.append({"phase": "phase4", "run_id": f"nsl_{tag}_s{s}", "seed": s, "split_seed": s,
                    "data": {"name": "nslkdd", "fs": "procedure"}, "mode": "fed", "train": S1["fl"], "fl": fl,
                    "detect": DET, "eval": {"test": True}})
    if os.path.exists(S3):
        t = json.load(open(S3))["nba"]
        for (sch, a), s in itertools.product([("dirichlet", 0.1), ("dirichlet", 0.5), ("dirichlet", 1.0)],  # natural = phase3 fedNBA_lam0.0
                                             range(5)):
            tag = sch if a is None else f"dir{a}"
            fl = {"n_clients": 45, "rounds": 15, "local_epochs": 4, "benign_scheme": sch, "attack_scheme": "group"}
            if a is not None:
                fl["benign_alpha"] = a
            out.append({"phase": "phase4", "run_id": f"nba_{tag}_s{s}", "seed": s, "split_seed": s,
                        "data": {"name": "nbaiot"}, "mode": "fed", "train": {**t, "batch_size": 256}, "fl": fl,
                        "detect": DET, "eval": {"test": True}})
    return out
