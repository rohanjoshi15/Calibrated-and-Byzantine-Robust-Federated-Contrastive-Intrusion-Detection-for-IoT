"""Final GPU re-run grid. Every config gets device='cuda'. STAGE (comma-separated):
  check     GPU determinism pair (corrC_s0 twice) + CPU-vs-GPU equivalence runs (results/phase8_check|phase8_equiv)
  tune_cic  CICIoT2023 re-tuning on the v2 split, seed 0, val only
  nsl       every NSL-KDD run used in results/tables, on the full 43-column test file; Byzantine and
            full-evaluation cells at 5 seeds
  nba       N-BaIoT Byzantine and full-evaluation cells at 5 seeds
  cic       CICIoT2023 central / FL / LOFO at 5 seeds on split v2
  h1        extra seeds 5-9 for the H1 paired study (NSL central+fed, CIC central)
  fedcic    federated CICIoT2023, seeds 0-9 (phase8/fedCIC_s0-9)
  cut_b     FedProx/MOON seeds 3-4 (first item to cut under a compute limit)
Run ids and result folders are those of the original grids, so the table scripts read them in place.
The queue is ordered longest-first (LPT) by a rough cost model; `cut_b` and federated CIC come last."""
import copy
import importlib.util
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def reseed(c, s):
    c = copy.deepcopy(c)
    c["seed"], c["split_seed"] = s, s
    c["run_id"] = re.sub(r"_s\d+$", f"_s{s}", c["run_id"])
    return c


def seeds(cfgs, ss):
    """configs generated for seed 0..k → same configs for every seed in ss (unique by run_id)."""
    base = {re.sub(r"_s\d+$", "", c["run_id"]): c for c in cfgs}
    return [reseed(c, s) for c in base.values() for s in ss]


def stage_check():
    P2 = _mod("phase2_grid")
    P3 = _mod("phase3_grid")
    P4 = _mod("phase4_grid")
    out = []
    c0 = next(c for c in P2.corrected() if c["run_id"] == "corrC_s0")
    for tag in ("A", "B"):
        out.append({**copy.deepcopy(c0), "phase": "phase8_check", "run_id": f"det{tag}_corrC_s0"})
    # CPU-vs-GPU equivalence: same config as an earlier CPU run, only the device differs (legacy k-means path).
    c = next(c for c in P2.leakage() if c["run_id"] == "leakK4_s1")
    out.append({**copy.deepcopy(c), "phase": "phase8_equiv", "run_id": "eqP2_leakK4_s1_mirror", "kmeans_impl": "legacy",
                "data": {**c["data"], "test_file": "mirror"}})
    c = next(c for c in P3.central() if c["run_id"] == "cenNBA_lam0.0_s0")
    out.append({**copy.deepcopy(c), "phase": "phase8_equiv", "run_id": "eqP3_cenNBA_lam0.0_s0", "kmeans_impl": "legacy"})
    c = next(c for c in P4.configs() if c["run_id"] == "nba_dir1.0_s4")
    out.append({**copy.deepcopy(c), "phase": "phase8_equiv", "run_id": "eqP4_nba_dir1.0_s4", "kmeans_impl": "legacy"})
    # Byzantine equivalence: stage 'nba' re-runs NBA_wl_f0.2_b1.0_fedavg_s1 with an identical config (K=1 → no k-means)
    return out


def stage_tune_cic():
    return _mod("phase6_grid").tune_cic()


def stage_nsl():
    P1, P2, P3, P4, P5, P6 = (_mod(f"phase{i}_grid") for i in range(1, 7))
    out = []
    for st in ("final_central", "final_fl", "debug_central"):
        out += getattr(P1, st)()
    out += P2.corrected() + P2.leakage() + P2.loco()
    out += [c for c in P3.central() if c["run_id"].startswith("cenNSL")]
    out += [c for c in P3.fed() if c["run_id"].startswith("fedNSL")]
    out += [c for c in P4.configs() if c["run_id"].startswith("nsl_")]
    out += seeds([c for c in P5.configs() if c["run_id"].startswith("NSL_")], range(5))
    out += seeds([c for c in P6.central() if c["run_id"].startswith("cen_NSL_")], range(5))
    out += _p6_fl("NSL", P6)
    return out


def _p6_fl(ds, P6, ss=range(5), cut=False):
    """Full-evaluation FL cells of dataset ds. cut=False: all algos seeds 0-2 + FedAvg seeds 3-4;
    cut=True: FedProx/MOON seeds 3-4 only (stage cut_b)."""
    base = [c for c in P6.fl() if c["run_id"].startswith(f"fl_{ds}_")]
    out = []
    for c in seeds(base, ss):
        alt = c["fl"].get("algo", "fedavg") != "fedavg"
        late = c["seed"] >= 3
        if (alt and late) == cut:
            out.append(c)
    return out


def stage_nba():
    P5, P6 = _mod("phase5_grid"), _mod("phase6_grid")
    out = seeds([c for c in P5.configs() if c["run_id"].startswith("NBA_")], range(5))
    out += seeds([c for c in P6.central() if c["run_id"].startswith("cen_NBA_")], range(5))
    out += _p6_fl("NBA", P6)
    out += seeds(P6.perdevice(), range(5))
    return out


def stage_cic():
    P6 = _mod("phase6_grid")
    out = seeds([c for c in P6.central() if c["run_id"].startswith("cen_CIC_")], range(5))
    out += _p6_fl("CIC", P6)
    out += seeds(P6.lofo(), range(5))
    return out


def stage_cut_b():
    P6 = _mod("phase6_grid")
    return _p6_fl("NSL", P6, cut=True) + _p6_fl("CIC", P6, cut=True)


def fed_cic(ss=range(10)):
    P6 = _mod("phase6_grid")
    t = P6.train("CIC", central=False)
    out = []
    for s in ss:
        out.append({"phase": "phase8", "run_id": f"fedCIC_s{s}", "seed": s, "split_seed": s, "data": {"name": "ciciot"},
                    "mode": "fed", "train": t,
                    "fl": {"n_clients": 50, "rounds": 15, "local_epochs": 4, "benign_scheme": "dirichlet",
                           "benign_alpha": 0.5, "attack_scheme": "dirichlet", "attack_alpha": 0.5},
                    "detect": {"K_fed": [1, 2, 4, 8, 16], "sigma": [0.0]}, "eval": {"test": True}})
    return out


def stage_h1():
    P3, P6 = _mod("phase3_grid"), _mod("phase6_grid")
    out = seeds([c for c in P3.central() if c["run_id"] == "cenNSL_lam0.0_s0"], range(5, 10))
    out += seeds([c for c in P3.fed() if c["run_id"] == "fedNSL_lam0.0_s0"], range(5, 10))
    out += seeds([c for c in P6.central() if c["run_id"] == "cen_CIC_feco_s0"], range(5, 10))
    return out


def cost(c):
    """Rough relative runtime (minutes, 6 concurrent cuda workers) for longest-first ordering."""
    ds = c["data"]["name"]
    mode = c.get("mode")
    meth = c.get("train", {}).get("loss", "feco")
    algo = c.get("fl", {}).get("algo", "fedavg")
    if mode == "iforest":
        return 0.5
    if ds == "nbaiot":
        if mode == "fed":
            return 20
        if mode == "fl":
            return 6
        if len(c["data"].get("devices", [])) == 1:
            return 1.5
        return 12 if meth in ("feco", "clad") else 4
    if ds == "ciciot":
        if mode in ("fl", "fed"):
            return (10 if meth in ("feco", "clad") else 4) * (1.4 if algo == "moon" else 1.0)
        return 4 if meth in ("feco", "clad") else 1.5
    if mode in ("fl", "fed", "self"):
        return (7 if meth in ("feco", "clad") else 4) * (1.4 if algo == "moon" else 1.0)
    return 4


def configs():
    st = os.environ.get("STAGE", "check").split(",")
    out = []
    tail = []
    for s in st:
        cs = globals()[f"stage_{s}"]() if s != "fedcic" else []
        (tail if s == "cut_b" else out).extend(cs)
        if s == "fedcic":
            tail.extend(fed_cic())
    out = sorted(out, key=cost, reverse=True) + sorted(tail, key=cost, reverse=True)
    seen, uniq = set(), []
    for c in out:
        k = (c["phase"], c["run_id"])
        if k not in seen:
            seen.add(k)
            c["device"] = "cuda"
            uniq.append(c)
    return uniq
