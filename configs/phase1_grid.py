"""FeCo reproduction grids (NSL-KDD). STAGE env var: tune_central | final_central | tune_fl | final_fl.
Selections are read from results/phase1 via val metrics only (src/utils/select.py)."""
import itertools
import json
import os

DATA = {"name": "nslkdd", "fs": "paper", "scaler": "minmax"}


def _sel(name):
    p = os.path.join("configs", "phase1_selected.json")
    return json.load(open(p))[name] if os.path.exists(p) else None


def tune_central():
    out = []
    for tau, mom, bs, ep in itertools.product([0.05, 0.1, 0.2, 0.5], [0.0, 0.9], [128, 512], [20, 60]):
        out.append({"phase": "phase1", "run_id": f"tuneC_tau{tau}_m{mom}_bs{bs}_ep{ep}", "seed": 0, "split_seed": 0,
                    "data": DATA, "mode": "central",
                    "train": {"tau": tau, "momentum": mom, "batch_size": bs, "epochs": ep, "normalize": True},
                    "eval": {"test": False}})
    # normalisation ablation (FeCo Eq. 2 literal: raw dot products)
    for tau, mom in itertools.product([0.1, 0.5, 1.0], [0.0, 0.9]):
        out.append({"phase": "phase1", "run_id": f"tuneC_nonorm_tau{tau}_m{mom}_bs512_ep60", "seed": 0, "split_seed": 0,
                    "data": DATA, "mode": "central",
                    "train": {"tau": tau, "momentum": mom, "batch_size": 512, "epochs": 60, "normalize": False},
                    "eval": {"test": False}})
    return out


def final_central():
    t = _sel("central")
    out = []
    for s in range(5):
        out.append({"phase": "phase1", "run_id": f"finalC_full_s{s}", "seed": s, "split_seed": s,
                    "data": {**DATA, "train_on": "full"}, "mode": "central", "train": t, "eval": {"test": True}})
    return out


def tune_fl():
    t = _sel("central")
    out = []
    for bs, mom in itertools.product([64, 128, 512], [0.0, 0.9]):
        out.append({"phase": "phase1", "run_id": f"tuneFL_iid_bs{bs}_m{mom}", "seed": 0, "split_seed": 0, "data": DATA,
                    "mode": "fl", "train": {**t, "batch_size": bs, "momentum": mom},
                    "fl": {"n_clients": 50, "scheme": "iid", "rounds": 15, "local_epochs": 4}, "eval": {"test": False}})
    return out


def final_fl():
    t = _sel("fl")
    out = []
    for mode, sch, s in itertools.product(["fl", "self"], ["iid", "noniid1", "noniid2"], range(5)):
        out.append({"phase": "phase1", "run_id": f"final{mode.upper()}_{sch}_full_s{s}", "seed": s, "split_seed": s,
                    "data": {**DATA, "train_on": "full"}, "mode": mode, "train": t,
                    "fl": {"n_clients": 50, "scheme": sch, "rounds": 15, "local_epochs": 4}, "eval": {"test": True}})
    return out


def debug_central():
    """Three pre-specified fidelity variants (not selected on test; all reported)."""
    t = _sel("central")
    A = {"A1_literal": ({}, {"tau": 0.1, "momentum": 0.0, "batch_size": 256, "epochs": 60, "attack_frac": "random"}),
         "A2_literal_zscore": ({"scaler": "zscore"}, {"tau": 0.1, "momentum": 0.0, "batch_size": 256, "epochs": 60,
                                                       "attack_frac": "random"}),
         "A3_sel_log1p": ({"log1p": True}, t)}
    out = []
    for name, (dx, tr) in A.items():
        for s in range(5):
            out.append({"phase": "phase1", "run_id": f"debugC_{name}_s{s}", "seed": s, "split_seed": s,
                        "data": {**DATA, "train_on": "full", **dx}, "mode": "central", "train": tr, "eval": {"test": True}})
    return out


def configs():
    return globals()[os.environ.get("STAGE", "tune_central")]()
