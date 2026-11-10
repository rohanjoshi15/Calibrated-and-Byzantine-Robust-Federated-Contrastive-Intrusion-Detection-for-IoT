"""In-process FL simulation: FedAvg / FedProx / MOON + robust aggregation + Byzantine hooks."""
import copy

import numpy as np
import torch

from ..models.train import train_local
from .aggregators import AGG, flatten, load_flat


def run_fl(model_fn, X, y, parts, tcfg, rounds=15, frac=1.0, agg="fedavg", agg_kw=None, seed=0, algo="fedavg",
           byzantine=None, round_cb=None, log=print):
    """X: torch float tensor (N,d) pooled; y: np array; parts: list of index arrays.
    algo ∈ {fedavg, fedprox, moon}: fedprox requires tcfg['mu_prox']>0, moon tcfg['mu_moon']>0.
    byzantine: optional object with .ids (set) and .local(cid, model, Xc, yc, tcfg, rng, global_model) -> model,
               and optional .post(cid, local_vec, global_vec) -> vec (update manipulation).
    round_cb(r, global_model) -> may return dict (e.g. protos for next round) merged into tcfg."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    g = model_fn()
    m = len(parts)
    prev = {}
    Xc = [X[torch.as_tensor(p, device=X.device)] for p in parts]
    yc = [y[p] for p in parts]
    hist = []
    extra = {}
    cfg = dict(tcfg)
    if algo == "fedavg":
        cfg["mu_prox"] = 0.0
        cfg["mu_moon"] = 0.0
    elif algo == "fedprox":
        cfg["mu_moon"] = 0.0
    elif algo == "moon":
        cfg["mu_prox"] = 0.0
    for r in range(rounds):
        sel = np.sort(rng.choice(m, max(1, int(round(frac * m))), replace=False))
        gvec = flatten(g)
        vecs, ns, losses = [], [], []
        for cid in sel:
            if (yc[cid] == 0).sum() < 2:
                continue
            local = copy.deepcopy(g)
            crng = np.random.RandomState(rng.randint(1 << 30))
            if byzantine is not None and cid in byzantine.ids:
                local = byzantine.local(cid, local, Xc[cid], yc[cid], {**cfg, **extra}, crng, g)
                info = {"loss": float("nan")}
            else:
                local, info = train_local(local, Xc[cid], yc[cid], {**cfg, **extra}, crng, global_model=g,
                                          prev_model=prev.get(cid), protos=extra.get("protos"))
            v = flatten(local)
            if byzantine is not None and cid in byzantine.ids and hasattr(byzantine, "post"):
                v = byzantine.post(cid, v, gvec)
            if cfg.get("mu_moon", 0) > 0:
                prev[cid] = local
            vecs.append(v)
            ns.append(len(yc[cid]))
            losses.append(info["loss"])
        W = torch.stack(vecs)
        newvec = AGG[agg](W, ns, **(agg_kw or {}))
        load_flat(g, newvec)
        rec = {"round": r, "loss": float(np.nanmean(losses)) if np.isfinite(losses).any() else None, "n_clients": len(vecs)}
        if round_cb is not None:
            out = round_cb(r, g)
            if out:
                extra.update({k: v for k, v in out.items() if k != "log"})
                rec.update(out.get("log", {}))
        hist.append(rec)
        log(f"round {r} loss {rec['loss']}")
    return g, hist


def run_self(model_fn, X, y, parts, tcfg, epochs, seed=0, log=print):
    """Self-learning: each client trains alone for `epochs` epochs. Returns list of models."""
    models = []
    for cid, p in enumerate(parts):
        torch.manual_seed(seed * 1000 + cid)
        mdl = model_fn()
        yy = y[p]
        if (yy == 0).sum() >= 2:
            mdl, _ = train_local(mdl, X[torch.as_tensor(p, device=X.device)], yy, {**tcfg, "epochs": epochs, "mu_prox": 0, "mu_moon": 0},
                                 np.random.RandomState(seed * 1000 + cid))
        models.append(mdl)
    return models
