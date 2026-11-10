"""Full evaluation. STAGE: tune_cic | tune_mu | central | fl | lofo | perdevice.
Compute cuts: 3 seeds; FedProx/MOON only for the contrastive family (FeCo, CLAD); SVDD/SAD/AE with FedAvg;
IsolationForest centralised only. Partitions: NSL-KDD {iid, dirichlet 0.5}; N-BaIoT {natural, dirichlet 0.5 over devices};
CICIoT2023 {iid, dirichlet 0.5 over benign capture files}. μ (FedProx) and μ_moon tuned on NSL-KDD val (seed 0).
FeCo-MP = FeCo encoder + K-prototype detector (K reported for every run via detect.K; the val-selected K is read in
analysis)."""
import itertools
import json
import os

S1 = json.load(open("configs/phase1_selected.json"))
S3 = json.load(open("configs/phase3_selected.json")) if os.path.exists("configs/phase3_selected.json") else {}
S6P = "configs/phase6_selected.json"
S6 = json.load(open(S6P)) if os.path.exists(S6P) else {}
SEEDS = range(3)
DET = {"K": [1, 2, 4, 8, 16], "vmf_K": [4, 8]}


def data(ds):
    return {"NSL": {"name": "nslkdd", "fs": "procedure"}, "NBA": {"name": "nbaiot"}, "CIC": {"name": "ciciot"}}[ds]


def train(ds, central=True):
    if ds == "NSL":
        return S1["central"] if central else S1["fl"]
    t = S3.get("nba") if ds == "NBA" else S6.get("cic")
    if t is None:
        return None
    return t if central else {**t, "batch_size": 256}


METHODS = {  # name -> train overrides
    "feco": {}, "clad": {"loss": "clad"}, "svdd": {"loss": "svdd", "attack_frac": 0.0},
    "sad": {"loss": "sad"}, "ae": {"loss": "ae", "attack_frac": 0.0},
}


def tune_cic():
    out = []
    for tau, ep in itertools.product([0.1, 0.2, 0.5], [10, 30]):
        out.append({"phase": "phase6", "run_id": f"tuneCIC_tau{tau}_ep{ep}", "seed": 0, "split_seed": 0,
                    "data": data("CIC"), "mode": "central",
                    "train": {"tau": tau, "momentum": 0.9, "batch_size": 512, "epochs": ep}, "detect": {"K": [1]},
                    "eval": {"test": False}})
    return out


def tune_mu():
    out = []
    fl = {"n_clients": 50, "rounds": 15, "local_epochs": 4, "benign_scheme": "dirichlet", "benign_alpha": 0.5,
          "attack_scheme": "dirichlet", "attack_alpha": 0.5}
    for mu in (0.01, 0.1, 1.0):
        out.append({"phase": "phase6", "run_id": f"tuneMU_fedprox_{mu}", "seed": 0, "split_seed": 0, "data": data("NSL"),
                    "mode": "fl", "train": {**S1["fl"], "mu_prox": mu}, "fl": {**fl, "algo": "fedprox"},
                    "detect": {"K": [1]}, "eval": {"test": False}})
    for mu in (0.1, 1.0, 5.0):
        out.append({"phase": "phase6", "run_id": f"tuneMU_moon_{mu}", "seed": 0, "split_seed": 0, "data": data("NSL"),
                    "mode": "fl", "train": {**S1["fl"], "mu_moon": mu}, "fl": {**fl, "algo": "moon"},
                    "detect": {"K": [1]}, "eval": {"test": False}})
    return out


def central():
    out = []
    for ds, s in itertools.product(("NSL", "NBA", "CIC"), SEEDS):
        t = train(ds)
        if t is None:
            continue
        for mname, ov in METHODS.items():
            out.append({"phase": "phase6", "run_id": f"cen_{ds}_{mname}_s{s}", "seed": s, "split_seed": s,
                        "data": data(ds), "mode": "central", "train": {**t, **ov},
                        "detect": DET if mname in ("feco", "clad") else {}, "eval": {"test": True}})
        out.append({"phase": "phase6", "run_id": f"cen_{ds}_iforest_s{s}", "seed": s, "split_seed": s, "data": data(ds),
                    "mode": "iforest", "eval": {"test": True}})
    return out


def parts(ds):
    if ds == "NSL":
        return {"iid": {"n_clients": 50, "benign_scheme": "iid", "attack_scheme": "iid"},
                "dir0.5": {"n_clients": 50, "benign_scheme": "dirichlet", "benign_alpha": 0.5,
                           "attack_scheme": "dirichlet", "attack_alpha": 0.5}}
    if ds == "NBA":
        return {"natural": {"n_clients": 45, "benign_scheme": "natural", "attack_scheme": "group"},
                "dir0.5": {"n_clients": 45, "benign_scheme": "dirichlet", "benign_alpha": 0.5, "attack_scheme": "group"}}
    return {"iid": {"n_clients": 50, "benign_scheme": "iid", "attack_scheme": "iid"},
            "dir0.5": {"n_clients": 50, "benign_scheme": "dirichlet", "benign_alpha": 0.5,
                       "attack_scheme": "dirichlet", "attack_alpha": 0.5}}


def fl():
    out = []
    mu_p, mu_m = S6.get("mu_prox", 0.1), S6.get("mu_moon", 1.0)
    for ds, s in itertools.product(("NSL", "NBA", "CIC"), SEEDS):
        t = train(ds, central=False)
        if t is None:
            continue
        dd = data(ds)
        pp = parts(ds)
        if ds == "NBA":  # compute cut: saturated dataset → natural partition, FedAvg, subsampled train
            pp = {"natural": pp["natural"]}
            dd = {"name": "nbaiot", "benign_frac": 0.25, "cap_train_att": 2000}
        for pname, p in pp.items():
            base_fl = {"rounds": 15, "local_epochs": 4, **p}
            for mname, ov in METHODS.items():
                algos = ("fedavg", "fedprox", "moon") if (mname in ("feco", "clad") and ds != "NBA") else ("fedavg",)
                for algo in algos:
                    tr = {**t, **ov}
                    if algo == "fedprox":
                        tr["mu_prox"] = mu_p
                    if algo == "moon":
                        tr["mu_moon"] = mu_m
                    out.append({"phase": "phase6", "run_id": f"fl_{ds}_{pname}_{mname}_{algo}_s{s}", "seed": s,
                                "split_seed": s, "data": dd, "mode": "fl", "train": tr,
                                "fl": {**base_fl, "algo": algo},
                                "detect": DET if mname in ("feco", "clad") else {}, "eval": {"test": True}})
    return out


def lofo():
    out = []
    t = train("CIC")
    if t is None:
        return out
    for fam, s in itertools.product(["DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"], SEEDS):
        out.append({"phase": "phase6", "run_id": f"lofo_CIC_{fam}_s{s}", "seed": s, "split_seed": s,
                    "data": {**data("CIC"), "holdout": fam}, "mode": "central", "train": t, "detect": DET,
                    "eval": {"test": True}})
    return out


def perdevice():
    """FeCo protocol per device (TDSC §V-E, Table VIII): one model per N-BaIoT device."""
    from src.data.nbaiot import DEVICES
    out = []
    t = train("NBA")
    if t is None:
        return out
    for dev, s in itertools.product(DEVICES, SEEDS):
        out.append({"phase": "phase6", "run_id": f"dev_{dev}_s{s}", "seed": s, "split_seed": s,
                    "data": {"name": "nbaiot", "devices": [dev]}, "mode": "central", "train": t, "detect": {"K": [1]},
                    "eval": {"test": True}})
    return out


def configs():
    st = os.environ.get("STAGE", "central")
    return sum((globals()[x]() for x in st.split(",")), [])
