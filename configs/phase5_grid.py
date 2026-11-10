"""Byzantine attacks and defences (C3). 3 seeds (compute cut).
Training-time: whitelisting model poisoning, frac ∈ {0.1,0.2,0.3} × agg ∈ {fedavg, trimmed(β=0.3), median} × boost ∈ {1,5};
clean references (attack 'none', frac 0.2 → same attacker ids, honest behaviour) per aggregator.
Post-training on the clean FedAvg model (byz_posthoc): prototype poisoning frac × {kmeans (none), geomed, trimmed} × K∈{1,4};
calibration poisoning frac (global histogram quantile vs Rob-FCP robust quantile, both computed)."""
import itertools
import json
import os

S1 = json.load(open("configs/phase1_selected.json"))
S3 = "configs/phase3_selected.json"
FR = [0.1, 0.2, 0.3]


def posthoc():
    out = []
    for f in FR:
        for K in (1, 4):
            for pa in ("kmeans", "geomed", "trimmed", "support"):
                out.append({"attack": "proto", "frac": f, "boost": 3.0, "K": K, "proto_agg": pa})
        out.append({"attack": "calib", "frac": f, "boost": 3.0, "f_max": 0.3})
    return out


def dsets():
    d = [("NSL", {"name": "nslkdd", "fs": "procedure"}, S1["fl"],
          {"n_clients": 50, "rounds": 15, "local_epochs": 4, "benign_scheme": "dirichlet", "benign_alpha": 0.5,
           "attack_scheme": "dirichlet", "attack_alpha": 0.5})]
    if os.path.exists(S3):
        d.append(("NBA", {"name": "nbaiot"}, {**json.load(open(S3))["nba"], "batch_size": 256},
                  {"n_clients": 45, "rounds": 15, "local_epochs": 4, "benign_scheme": "natural", "attack_scheme": "group"}))
    return d


def configs():
    out = []
    for (ds, data, tr, fl), s in itertools.product(dsets(), range(3)):
        for agg in ("fedavg", "trimmed", "median"):
            kw = {"beta": 0.3} if agg == "trimmed" else None
            base = {"phase": "phase5", "seed": s, "split_seed": s, "data": data, "mode": "fed", "train": tr,
                    "fl": {**fl, "agg": agg, "agg_kw": kw}, "detect": {"K_fed": [1]}, "eval": {"test": True}}
            c = {**base, "run_id": f"{ds}_clean_{agg}_s{s}", "byz": {"attack": "none", "frac": 0.2}}
            if agg == "fedavg":
                c["byz_posthoc"] = posthoc()
            out.append(c)
            for f, b in itertools.product(FR, (1.0, 5.0)):
                out.append({**base, "run_id": f"{ds}_wl_f{f}_b{b}_{agg}_s{s}",
                            "byz": {"attack": "whitelist", "frac": f, "boost": b, "atk_lr_mult": 10.0}})
    return out
