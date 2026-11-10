"""Config-driven experiment pipeline shared by all phases."""
import copy
import hashlib
import json
import os
import pickle

import numpy as np
import torch

from .calib import thresholds as th
from .data import datasets as ds
from .data.partition import partition, describe
from .eval import metrics as M
from .fl.server import run_fl, run_self
from .models import detect as det
from .models.encoder import MLP, embed
from .models.train import train_local
from .utils import common as U
from .utils.common import set_seed

_DATA_MEM = {}


def data_version(dcfg):
    """Identity of the raw inputs / split protocol that the config alone does not pin down.
    nslkdd: md5 of the raw test file actually read; ciciot: split-protocol version. N-BaIoT: None (unchanged,
    so its key and cache stay identical)."""
    if dcfg["name"] == "nslkdd":
        from .data import nslkdd as nk
        return {"test_file": dcfg.get("test_file", "full"), "test_md5": nk.test_file_md5(dcfg.get("test_file", "full"))}
    if dcfg["name"] == "ciciot":
        return {"split": ds.CICIOT_SPLIT_VERSION}
    return None


def cache_key(dcfg, seed):
    v = data_version(dcfg)
    return json.dumps({**dcfg, "seed": seed, **({"_data_version": v} if v else {})}, sort_keys=True), v


def build_data(dcfg, seed):
    key, ver = cache_key(dcfg, seed)
    if key in _DATA_MEM:
        return _DATA_MEM[key]
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    path = os.path.join(ds.CACHE, f"{dcfg['name']}_{h}.pkl")
    if os.path.exists(path):
        D = pickle.load(open(path, "rb"))
    else:
        kw = {k: v for k, v in dcfg.items() if k != "name"}
        D = getattr(ds, dcfg["name"])(seed=seed, **kw)
        D.pop("frames", None)
        os.makedirs(ds.CACHE, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        pickle.dump(D, open(tmp, "wb"), protocol=4)
        os.replace(tmp, path)
    # fixed-width unicode label arrays cost ~4 bytes/char/row (≈160 B/row for N-BaIoT device names); object arrays
    # share the Python strings (8 B/row). Same values and comparisons.
    for sp in ("train", "val", "calib", "test", "val_all", "zeroday"):
        if sp in D:
            for f in ("cat", "group", "label"):
                if f in D[sp] and getattr(D[sp][f], "dtype", None) is not None and D[sp][f].dtype.kind == "U":
                    D[sp][f] = D[sp][f].astype(object)
    D["data_version"] = ver
    _DATA_MEM.clear()
    _DATA_MEM[key] = D
    return D


def fit_detectors(Zb, dcfg, seed, w=None):
    """Zb: normalised benign train embeddings. Returns {name: (kind, params)}."""
    out = {"template": ("template", det.template(Zb))}
    for K in dcfg.get("K", []):
        if K == 1:
            continue
        C, cnt = det.spherical_kmeans(Zb, K, w=w, seed=seed)
        out[f"kmeans{K}"] = ("maxcos", C)
    for K in dcfg.get("vmf_K", []):
        out[f"vmf{K}"] = ("vmf", det.fit_vmf_mixture(Zb, K, seed=seed, w=w))
    return out


def score(kind, p, Z):
    if kind == "template":
        return det.score_template(Z, p)
    if kind == "maxcos":
        return det.score_maxcos(Z, p)
    if kind == "vmf":
        return det.score_vmf(Z, p)
    raise ValueError(kind)


def subsample(n, cap, seed):
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.RandomState(seed).choice(n, cap, replace=False))


NATIVE = ("svdd", "sad", "ae", "iforest")


def make_scorers(model, D, cfg, seed, benign_train_idx=None):
    """Returns (scorers, s_train_benign) where scorers: {name: f(X)->similarity score (higher=benign)}."""
    ecfg = cfg.get("detect", {})
    meth = cfg.get("train", {}).get("loss", "feco") if cfg.get("mode") != "iforest" else "iforest"
    tr = D["train"]
    ib = np.flatnonzero(tr["y"] == 0) if benign_train_idx is None else benign_train_idx
    ib = ib[subsample(len(ib), ecfg.get("max_train_benign", 200000), seed)]
    Xb = tr["X"][ib]
    if meth in NATIVE:
        if meth == "iforest":
            f = lambda X: model.score_samples(X)  # noqa: higher = more normal
        elif meth == "ae":
            def f(X):
                with torch.no_grad():
                    Xt = torch.as_tensor(X)
                    dev = next(model.parameters()).device
                    out = []
                    for i in range(0, len(Xt), 8192):
                        xb = Xt[i:i + 8192].to(dev)
                        out.append(((model(xb) - xb) ** 2).sum(1).cpu().numpy())
                    return -np.concatenate(out) if out else np.zeros(0)
        else:
            cen = np.asarray(cfg["train"]["center"], np.float32)
            f = lambda X: -((embed(model, X, normalize=False) - cen) ** 2).sum(1)  # noqa
        return {"native": f}, {"native": f(Xb)}
    Zb = embed(model, Xb, normalize=True)
    dets = fit_detectors(Zb, ecfg, seed)
    scorers = {n: (lambda X, k=k, p=p: score(k, p, embed(model, X, normalize=True))) for n, (k, p) in dets.items()}
    return scorers, {n: score(k, p, Zb) for n, (k, p) in dets.items()}


def evaluate_scores(D, cfg, scorers, s_train, splits=("val", "test")):
    p = cfg.get("thresholds", {}).get("p", 5.0)
    alphas = cfg.get("thresholds", {}).get("alphas", [0.01, 0.05, 0.1])
    do_test = cfg.get("eval", {}).get("test", False)
    # with train_on='full' the NSL-KDD calib rows are inside the training set, so split-conformal
    # thresholds are not valid; they are stored under 'invalid_conf_*' and no table reads them.
    cpre = "invalid_conf" if cfg.get("data", {}).get("train_on") == "full" else "conf"
    res = {}
    for name, f in scorers.items():
        s_cal = f(D["calib"]["X"])
        rho = {"feco_p%g" % p: th.feco_percentile(s_train[name], p)}
        for a in alphas:
            rho[f"{cpre}_a{a}"] = th.conformal(s_cal, a)
        r = {"rho": rho}
        for split in splits:
            if split in ("test", "zeroday") and not do_test:
                continue
            s = f(D[split]["X"])
            d = D[split]
            rk = M.ranking(d["y"], s)
            r[split] = {"ranking": rk}
            for tn, rv in rho.items():
                r[split][tn] = M.evaluate(d["y"], s, rv, groups=d["cat"], novel=d.get("novel"), rank=rk)
        res[name] = r
    return res


def evaluate_model(model, D, cfg, seed, benign_train_idx=None):
    scorers, s_tr = make_scorers(model, D, cfg, seed, benign_train_idx)
    splits = ("val", "test") + (("zeroday",) if "zeroday" in D else ())
    return evaluate_scores(D, cfg, scorers, s_tr, splits)


def build_model_fn(cfg, D, seed, X, y):
    """Returns (model_fn, tcfg) — handles baselines (bias-free SVDD/SAD net + centre, AE)."""
    tcfg = dict(cfg.get("train", {}))
    meth = tcfg.get("loss", "feco")
    out = cfg.get("model", {}).get("out", 128)
    dev = U.DEV  # weights are initialised on CPU (same init as the CPU runs for a given seed), then moved to the device
    if meth == "ae":
        from .models.encoder import AE
        return (lambda: AE(D["dim"], out=out).to(dev)), tcfg
    if meth in ("svdd", "sad"):
        fn = lambda: MLP(D["dim"], out=out, bias=False).to(dev)  # noqa (Ruff et al.: no bias terms)
        if "center" not in tcfg:
            torch.manual_seed(seed)
            m0 = fn()
            with torch.no_grad():
                c = m0(X[torch.as_tensor(np.flatnonzero(y == 0), device=X.device)]).mean(0).cpu().numpy()
            eps = 0.1
            c[(np.abs(c) < eps) & (c < 0)] = -eps
            c[(np.abs(c) < eps) & (c >= 0)] = eps
            tcfg["center"] = c.tolist()
            cfg.setdefault("train", {})["center"] = tcfg["center"]
        return fn, tcfg
    return (lambda: MLP(D["dim"], out=out).to(dev)), tcfg


def federated_drop(cfg, seed, split_seed):
    """Federated FeCo feature selection: each client computes Spearman/MI/ANOVA/variance on its own
    partition of the NSL-KDD train split; server takes the count-weighted average and applies the FeCo rules.
    Partition is identical to the one used for training (same y/cat arrays, same seed)."""
    from .data import nslkdd as nk
    tr, _ = nk.load_raw()
    sp = np.load(os.path.join(nk.OUT, f"splits_seed{split_seed}.npz"))
    i_tr = np.arange(len(tr)) if cfg["data"].get("train_on") == "full" else sp["train"]
    ho = cfg["data"].get("holdout")
    if ho is not None:
        i_tr = i_tr[tr.category.values[i_tr] != ho]
    dtr = tr.iloc[i_tr]
    f = cfg["fl"]
    parts = partition(dtr.y.values, dtr.category.values, f.get("n_clients", 50), f.get("scheme", "iid"), seed=seed,
                      alpha=f.get("alpha", 0.5), no_attack_frac=f.get("no_attack_frac", 0.0))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = [nk.local_stats(dtr.iloc[p], dtr.category.values[p]) for p in parts]
    drop, info = nk.select_features(nk.aggregate_stats(stats))
    return drop, info


def run(cfg, log=print):
    seed = cfg.get("seed", 0)
    U.set_device(cfg.get("device", "cpu"))
    det.KMEANS_IMPL = cfg.get("kmeans_impl", "fast")
    set_seed(seed)
    torch.set_num_threads(cfg.get("threads", 1))
    fs_info = None
    if cfg["data"].get("fs") == "federated":
        drop, fs_info = federated_drop(cfg, seed, cfg.get("split_seed", seed))
        cfg = {**cfg, "data": {**{k: v for k, v in cfg["data"].items() if k != "fs"}, "drop": sorted(drop)}}
    D = build_data(cfg["data"], cfg.get("split_seed", seed))
    X = torch.as_tensor(D["train"]["X"]).to(U.DEV)
    y = D["train"]["y"]
    mode = cfg.get("mode", "central")
    model_fn, tcfg = build_model_fn(cfg, D, seed, X, y)
    out = {"dim": int(D["dim"]), "n_train": int(len(y)), "n_train_benign": int((y == 0).sum()),
           "dropped_features": D.get("drop"), "fs_info": fs_info, **U.device_info(),
           "data_version": D.get("data_version"), "kmeans_impl": det.KMEANS_IMPL}
    if mode == "iforest":
        out["device"] = "cpu-sklearn"  # IsolationForest has no GPU path (sklearn); deterministic given the seed
        from sklearn.ensemble import IsolationForest
        ib = np.flatnonzero(y == 0)
        mdl = IsolationForest(n_estimators=cfg.get("iforest", {}).get("n_estimators", 100), random_state=seed,
                              n_jobs=1).fit(D["train"]["X"][ib])
        out["results"] = evaluate_model(mdl, D, cfg, seed)
    elif mode == "central":
        set_seed(seed)
        model = model_fn()
        cb, P0 = None, None
        if tcfg.get("lam_proto", 0) > 0:
            ibs = np.flatnonzero(y == 0)
            ibs = ibs[subsample(len(ibs), 20000, seed)]
            Kt = tcfg.get("K_train", 4)

            def cb(m):  # prototypes refreshed each epoch
                return det.spherical_kmeans(embed(m, D["train"]["X"][ibs]), Kt, seed=seed, n_init=1, iters=30)[0]
            P0 = cb(model)
        model, info = train_local(model, X, y, tcfg, np.random.RandomState(seed), protos=P0, epoch_cb=cb)
        out["train_info"] = info
        out["results"] = evaluate_model(model, D, cfg, seed)
    elif mode in ("fl", "self"):
        f = cfg["fl"]
        if "benign_scheme" in f:  # group-based client builder (same as mode 'fed'), pooled evaluation
            from .fl.fedeval import build_clients
            parts = build_clients(D, f, seed)["train"]
        else:
            parts = partition(y, D["train"]["cat"], f.get("n_clients", 50), f.get("scheme", "iid"), seed=seed,
                              alpha=f.get("alpha", 0.5), no_attack_frac=f.get("no_attack_frac", 0.0))
        out["partition"] = describe(parts, y, D["train"]["cat"])
        if mode == "fl":
            model, hist = run_fl(model_fn, X, y, parts, {**tcfg, "epochs": f.get("local_epochs", 4)},
                                 rounds=f.get("rounds", 15), frac=f.get("frac", 1.0), agg=f.get("agg", "fedavg"),
                                 agg_kw=f.get("agg_kw"), seed=seed, algo=f.get("algo", "fedavg"), log=log)
            out["history"] = hist
            out["results"] = evaluate_model(model, D, cfg, seed)
        else:
            models = run_self(model_fn, X, y, parts, tcfg, epochs=f.get("rounds", 15) * f.get("local_epochs", 4),
                              seed=seed, log=log)
            per = []
            for cid, (mdl, p) in enumerate(zip(models, parts)):
                ibc = p[y[p] == 0]
                c2 = copy.deepcopy(cfg)
                c2.setdefault("detect", {})["K"] = []
                c2["detect"]["vmf_K"] = []
                r = evaluate_model(mdl, D, c2, seed, benign_train_idx=ibc)
                per.append(r[list(r)[0]])
            out["results_per_client"] = per
            out["results"] = {list(r)[0]: average_results(per)}
    elif mode == "fed":
        out.update(run_fed(cfg, D, X, y, model_fn, tcfg, seed, log))
    else:
        raise ValueError(mode)
    return out


def run_fed(cfg, D, X, y, model_fn, tcfg, seed, log):
    from .fl import fedeval as fe
    f = cfg["fl"]
    C = fe.build_clients(D, f, seed)
    parts = C["train"]
    ben_parts = [p[y[p] == 0] for p in parts]
    out = {"partition": describe(parts, y, D["train"]["cat"]), "share_groups": C["groups"]}
    byz = None
    if cfg.get("byz"):
        from .attacks.byzantine import make_byzantine
        byz = make_byzantine(cfg["byz"], D, C, seed)
        out["byz_ids"] = sorted(int(i) for i in byz.ids)
    cb = None
    if tcfg.get("lam_proto", 0) > 0:
        Kt = tcfg.get("K_train", 4)

        def cb(r, g):
            P, _, _ = fe.federated_prototypes(g, X, ben_parts, Kt, seed + r)
            return {"protos": P}
    model, hist = run_fl(model_fn, X, y, parts, {**tcfg, "epochs": f.get("local_epochs", 4)},
                         rounds=f.get("rounds", 15), frac=f.get("frac", 1.0), agg=f.get("agg", "fedavg"),
                         agg_kw=f.get("agg_kw"), seed=seed, algo=f.get("algo", "fedavg"), byzantine=byz,
                         round_cb=cb, log=log)
    out["history"] = hist
    ec = cfg.get("detect", {})
    res = {}
    for K in ec.get("K_fed", [1]):
        for sigma in ec.get("sigma", [0.0]):
            if byz is not None:
                byz.set_model(model)
            poison = byz.proto_poison if byz is not None else None
            P, Cs, cnts = fe.federated_prototypes(model, X, ben_parts, K, seed, sigma=sigma, poison=poison)
            if byz is not None and hasattr(byz, "robust_protos") and ec.get("proto_agg", "kmeans") != "kmeans":
                P = byz.robust_protos(Cs, cnts, K, ec["proto_agg"])
            cp = byz.calib_poison if (byz is not None and hasattr(byz, "calib_poison")) else None
            r = fe.per_client_eval(model, D, C, P, alphas=cfg.get("thresholds", {}).get("alphas", [0.01, 0.05, 0.1]),
                                   calib_poison=cp, f_max=cfg.get("byz", {}).get("f_max", 0.3) if cfg.get("byz") else 0.3,
                                   calib_caps=ec.get("calib_caps", []) if (K == 1 and sigma == 0) else [], seed=seed)
            if byz is not None:
                r["byz"] = byz.evaluate(model, P, D, C)
            res[f"K{K}_sigma{sigma}"] = r
            log(f"K={K} sigma={sigma} auroc={r['ranking_test'].get('auroc')}")
            if sigma == 0 and byz is None and ec.get("fed_vmf", True):
                from .models import detect as det_
                vm = fe.federated_vmf(model, X, ben_parts, P)
                rv = fe.per_client_eval(model, D, C, None, alphas=cfg.get("thresholds", {}).get("alphas", [0.01, 0.05, 0.1]),
                                        score_fn=lambda Z, vm=vm: det_.score_vmf(Z, vm), seed=seed, hist_ok=False)
                res[f"vmfK{K}"] = {k: v for k, v in rv.items()}
                log(f"vmf K={K} auroc={rv['ranking_test'].get('auroc')}")
    out["fed_results"] = res
    # post-training Byzantine attacks (prototype / calibration poisoning) evaluated on the same trained model
    from .attacks.byzantine import make_byzantine
    alphas = cfg.get("thresholds", {}).get("alphas", [0.01, 0.05, 0.1])
    posthoc = {}
    for spec in cfg.get("byz_posthoc", []):
        b = make_byzantine(spec, D, C, seed)
        b.set_model(model)
        K = spec.get("K", 1)
        P, Cs, cnts = fe.federated_prototypes(model, X, ben_parts, K, seed, poison=b.proto_poison)
        if spec.get("proto_agg", "kmeans") != "kmeans":
            P = b.robust_protos(Cs, cnts, K, spec["proto_agg"])
        r = fe.per_client_eval(model, D, C, P, alphas=alphas, calib_poison=b.calib_poison, f_max=spec.get("f_max", 0.3))
        r = {k: ({kk: vv for kk, vv in v.items() if not kk.endswith("per_client")} if isinstance(v, dict) else v)
             for k, v in r.items() if k not in ("calib_n", "test_benign_n")}
        r["byz"] = b.evaluate(model, P, D, C)
        if getattr(b, "last_support", None):
            r["byz"]["support"] = b.last_support
        name = "_".join(f"{k}{v}" for k, v in spec.items())
        posthoc[name] = r
        log(f"posthoc {name}: auroc={r['ranking_test'].get('auroc')} byz={r['byz']}")
    if posthoc:
        out["byz_posthoc"] = posthoc
    return out


def average_results(per):
    """Average scalar metrics over clients (FeCo Table VII self-learning: mean over clients)."""
    def rec(objs):
        o0 = objs[0]
        if isinstance(o0, dict):
            return {k: rec([o[k] for o in objs if k in o]) for k in o0}
        if isinstance(o0, (int, float, np.floating)) and not isinstance(o0, bool):
            return float(np.nanmean([float(o) for o in objs]))
        return o0
    return rec(per)
