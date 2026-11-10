"""Multi-prototype detectors (C1). STAGE: tune_nba | central | fed.
NSL-KDD uses the corrected protocol (fs=procedure on the 80% split; val-selected hyper-parameters).
N-BaIoT: one pooled model across 9 devices (cross-device); hyper-parameters tuned on val (seed 0) in tune_nba.
Post-hoc detectors on the same trained encoder: K∈{1,2,4,8,16} spherical k-means (max-cos), vMF mixtures K∈{1,4,8}.
λ (prototype compactness loss, K_train=4, prototypes refreshed each epoch/round) ∈ {0, 0.1, 0.5}."""
import itertools
import json
import os

S1 = json.load(open("configs/phase1_selected.json"))
SEL = "configs/phase3_selected.json"
DET = {"K": [1, 2, 4, 8, 16], "vmf_K": [1, 4, 8]}
NSL = {"name": "nslkdd", "fs": "procedure"}
NBA = {"name": "nbaiot"}


def _nba():
    return json.load(open(SEL))["nba"] if os.path.exists(SEL) else None


def tune_nba():
    out = []
    for tau, ep in itertools.product([0.1, 0.2, 0.5], [10, 30]):
        out.append({"phase": "phase3", "run_id": f"tuneNBA_tau{tau}_ep{ep}", "seed": 0, "split_seed": 0, "data": NBA,
                    "mode": "central", "train": {"tau": tau, "momentum": 0.9, "batch_size": 512, "epochs": ep},
                    "detect": {"K": [1]}, "eval": {"test": False}})
    return out


def central():
    out = []
    for lam, s in itertools.product([0.0, 0.1, 0.5], range(5)):
        out.append({"phase": "phase3", "run_id": f"cenNSL_lam{lam}_s{s}", "seed": s, "split_seed": s, "data": NSL,
                    "mode": "central", "train": {**S1["central"], "lam_proto": lam, "K_train": 4}, "detect": DET,
                    "eval": {"test": True}})
        t = _nba()
        if t is not None:
            out.append({"phase": "phase3", "run_id": f"cenNBA_lam{lam}_s{s}", "seed": s, "split_seed": s, "data": NBA,
                        "mode": "central", "train": {**t, "lam_proto": lam, "K_train": 4}, "detect": DET,
                        "eval": {"test": True}})
    return out


FEDDET = {"K_fed": [1, 2, 4, 8, 16], "sigma": [0.0, 0.01, 0.05, 0.1], "calib_caps": [25, 50, 100, 200]}


def fed():
    out = []
    for lam, s in itertools.product([0.0, 0.1], range(5)):
        out.append({"phase": "phase3", "run_id": f"fedNSL_lam{lam}_s{s}", "seed": s, "split_seed": s, "data": NSL,
                    "mode": "fed", "train": {**S1["fl"], "lam_proto": lam, "K_train": 4},
                    "fl": {"n_clients": 50, "rounds": 15, "local_epochs": 4, "benign_scheme": "dirichlet",
                           "benign_alpha": 0.5, "attack_scheme": "dirichlet", "attack_alpha": 0.5},
                    "detect": FEDDET, "eval": {"test": True}})
        t = _nba()
        if t is not None:
            out.append({"phase": "phase3", "run_id": f"fedNBA_lam{lam}_s{s}", "seed": s, "split_seed": s, "data": NBA,
                        "mode": "fed", "train": {**t, "batch_size": 256, "lam_proto": lam, "K_train": 4},
                        "fl": {"n_clients": 45, "rounds": 15, "local_epochs": 4, "benign_scheme": "natural",
                               "attack_scheme": "group"},
                        "detect": FEDDET, "eval": {"test": True}})
    return out


def configs():
    st = os.environ.get("STAGE", "tune_nba")
    return sum((globals()[x]() for x in st.split(",")), [])
