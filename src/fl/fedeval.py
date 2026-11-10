"""Federated training + per-client calibration/evaluation.

Client data assignment: every split's rows carry a `group` (benign heterogeneity key). A client×group share matrix P
(columns sum to 1) is drawn once and used for train/calib/val/test benign alike, so each client's calibration,
in-distribution hold-out and test benign come from the same mixture:
  benign_scheme 'natural'  : n_clients = G·m; group g is spread evenly over its m own clients (N-BaIoT: device×gateways)
                'dirichlet': P[:, g] ~ Dir(benign_alpha·1)
                'iid'      : P[:, g] = 1/n
Train attacks: attack_scheme 'group' (follow the group of the attack row, N-BaIoT natural), or iid / noniid1 /
noniid2 / dirichlet over attack categories (src.data.partition). Test attacks: by group if attack_scheme='group',
else the full test attack set is shared by all clients.
Prototypes: each client runs spherical k-means (K) on its benign train embeddings → (centroids, counts) [+ Gaussian
noise σ on centroids, renormalised] → server weighted spherical k-means → K global prototypes (K=1: count-weighted mean).
Thresholds per client i: (a) feco_global: p-percentile of pooled benign train scores; feco_local: of own train scores;
(b) conformal_local: split-conformal on own calib; (c) conformal_global: federated histogram quantile of all calib;
(d) robust_global: Rob-FCP histogram filtering (only meaningful under Byzantine calibration reports)."""
import math

import numpy as np
import torch

from ..calib import thresholds as th
from ..calib.robust import robust_quantile
from ..data.partition import partition
from ..eval import metrics as M
from ..models import detect as det
from ..models.encoder import embed


def share_matrix(groups, n_clients, scheme, alpha, rng):
    G = list(groups)
    if scheme == "natural":
        assert n_clients % len(G) == 0, "natural scheme needs n_clients multiple of #groups"
        m = n_clients // len(G)
        P = np.zeros((n_clients, len(G)))
        for gi in range(len(G)):
            P[gi * m:(gi + 1) * m, gi] = 1.0 / m
    elif scheme == "dirichlet":
        P = rng.dirichlet(alpha * np.ones(n_clients), size=len(G)).T
    elif scheme == "iid":
        P = np.full((n_clients, len(G)), 1.0 / n_clients)
    else:
        raise ValueError(scheme)
    return P, G


def assign(groups_arr, idx, P, G, rng):
    """Split row indices `idx` (all from one split) among clients according to P[:, group]."""
    parts = [[] for _ in range(P.shape[0])]
    for gi, g in enumerate(G):
        ig = rng.permutation(idx[groups_arr[idx] == g])
        if len(ig) == 0:
            continue
        cuts = np.round(np.cumsum(P[:, gi]) * len(ig)).astype(int)[:-1]
        for c, s in enumerate(np.split(ig, cuts)):
            parts[c].append(s)
    return [np.sort(np.concatenate(p)) if p else np.array([], int) for p in parts]


def build_clients(D, f, seed):
    rng = np.random.RandomState(seed + 12345)
    n = f.get("n_clients", 50)
    tr = D["train"]
    bgroups = sorted(set(tr["group"][tr["y"] == 0]))
    P, G = share_matrix(bgroups, n, f.get("benign_scheme", "iid"), f.get("benign_alpha", 0.5), rng)
    out = {"P": P, "groups": G}
    ib = np.flatnonzero(tr["y"] == 0)
    ben = assign(tr["group"], ib, P, G, rng)
    ia = np.flatnonzero(tr["y"] == 1)
    asch = f.get("attack_scheme", "iid")
    if asch == "group":
        att = assign(tr["group"], ia, P, G, rng)
    else:
        sub = partition(tr["y"][ia], tr["cat"][ia], n, asch, seed=seed, alpha=f.get("attack_alpha", 0.5),
                        no_attack_frac=f.get("no_attack_frac", 0.0))
        att = [ia[s] for s in sub]
    out["train"] = [np.sort(np.r_[b, a]) for b, a in zip(ben, att)]
    for split in ("calib", "val", "test"):
        d = D[split]
        out[split + "_benign"] = assign(d["group"], np.flatnonzero(d["y"] == 0), P, G, rng)
    te = D["test"]
    ta = np.flatnonzero(te["y"] == 1)
    out["test_attack"] = assign(te["group"], ta, P, G, rng) if asch == "group" else [ta] * n
    return out


def client_centroids(Zb, K, seed, sigma=0.0, rng=None):
    if len(Zb) == 0:
        return np.zeros((0, Zb.shape[1])), np.zeros(0)
    if K == 1:
        C, cnt = det.unit(Zb.mean(0, keepdims=True)), np.array([len(Zb)], float)
    else:
        C, cnt = det.spherical_kmeans(Zb, min(K, len(Zb)), seed=seed, n_init=1, iters=30)
    if sigma > 0:
        C = det.unit(C + (rng or np.random).normal(0, sigma, C.shape))
    return C, cnt


def server_prototypes(Cs, cnts, K, seed):
    C = np.concatenate([c for c in Cs if len(c)])
    w = np.concatenate([c for c in cnts if len(c)])
    if K == 1:
        return det.unit((C * w[:, None]).sum(0, keepdims=True))
    P, _ = det.spherical_kmeans(C, min(K, len(C)), w=w, seed=seed, n_init=3)
    return P


def federated_prototypes(model, X, parts_benign, K, seed, sigma=0.0, poison=None):
    """parts_benign: list of benign-train index arrays. poison(cid, C, cnt) -> (C, cnt) for Byzantine clients."""
    rng = np.random.RandomState(seed + 999)
    Cs, cnts = [], []
    for cid, p in enumerate(parts_benign):
        Zb = embed(model, X[torch.as_tensor(p, device=X.device)], normalize=True) if len(p) else np.zeros((0, 1))
        C, cnt = client_centroids(Zb, K, seed + cid, sigma, rng)
        if poison is not None:
            C, cnt = poison(cid, C, cnt)
        Cs.append(C)
        cnts.append(cnt)
    return server_prototypes(Cs, cnts, K, seed), Cs, cnts


def federated_vmf(model, X, parts_benign, P, kappa_max=1e4):
    """One extra round after prototypes P (K×o): each client assigns its benign embeddings to the nearest prototype and
    sends per-prototype resultant sums and counts; server sets μ_k = S_k/‖S_k‖, κ_k = Banerjee(‖S_k‖/N_k), π_k ∝ N_k
    (one federated EM step for a vMF mixture)."""
    K, d = P.shape
    S, N = np.zeros((K, d)), np.zeros(K)
    for p in parts_benign:
        if len(p) == 0:
            continue
        Z = embed(model, X[torch.as_tensor(p, device=X.device)], normalize=True)
        a = np.argmax(Z @ P.T, 1)
        if det.KMEANS_IMPL == "legacy":
            np.add.at(S, a, Z)  # legacy path (bit-exact with the earlier CPU runs)
        else:
            S += det.scatter_sum(a, Z, K)
        N += np.bincount(a, minlength=K)
    keep = N > 0
    S, N = S[keep], N[keep]
    norm = np.linalg.norm(S, axis=1)
    return {"mu": S / norm[:, None], "kappa": np.minimum(det.kappa_banerjee(norm / N, d), kappa_max), "pi": N / N.sum()}


def per_client_eval(model, D, C, protos, alphas=(0.01, 0.05, 0.1), p=5.0, calib_poison=None, f_max=0.3,
                    score_fn=None, calib_caps=(), seed=0, hist_ok=True):
    """protos: (K, o) global prototypes; score = max cos. calib_poison(cid, scores)->scores for Byzantine reports
    (only affects what the server sees for (c)/(d); local conformal thresholds of honest clients are unaffected)."""
    X = D["train"]["X"]
    sf = score_fn or (lambda Z: det.score_maxcos(Z, protos))
    n = len(C["train"])

    def scores(split, idx):
        return sf(embed(model, D[split]["X"][idx], normalize=True)) if len(idx) else np.zeros(0)

    tr_b = [p[D["train"]["y"][p] == 0] for p in C["train"]]
    s_tr = [scores("train", p) for p in tr_b]
    s_cal = [scores("calib", p) for p in C["calib_benign"]]
    s_id = [scores("val", p) for p in C["val_benign"]]
    s_teb = [scores("test", p) for p in C["test_benign"]]
    # test attacks: score once if shared
    shared = all(a is C["test_attack"][0] for a in C["test_attack"])
    s_all_att = scores("test", C["test_attack"][0]) if shared else None
    s_tea = [s_all_att if shared else scores("test", a) for a in C["test_attack"]]

    reported = [calib_poison(i, s) if calib_poison else s for i, s in enumerate(s_cal)]
    hists = [th.histogram(s) for s in reported] if hist_ok else None
    res = {"n_clients": n, "calib_n": [len(s) for s in s_cal], "test_benign_n": [len(s) for s in s_teb],
           "id_n": [len(s) for s in s_id]}
    methods = {}
    for a in alphas:
        methods[f"conformal_local_a{a}"] = [th.conformal(s, a) for s in s_cal]
        if hist_ok:  # histogram methods need scores in [-1, 1] (cosine); not used for vMF log-likelihoods
            rg = th.from_histogram(sum(hists), a)
            rr, keep, _ = robust_quantile(hists, a, f_max=f_max)
            methods[f"conformal_global_a{a}"] = [rg] * n
            methods[f"robust_global_a{a}"] = [rr] * n
        methods[f"conformal_local_ceil_a{a}"] = [th.conformal(s, a, "ceil") for s in s_cal]
        # FeCo thresholds at the same nominal level p = 100·α
        methods[f"feco_global_a{a}"] = [th.feco_percentile(np.concatenate(s_tr), 100 * a)] * n
        methods[f"feco_local_a{a}"] = [th.feco_percentile(s, 100 * a) if len(s) else -np.inf for s in s_tr]
        # calibration-size ablation: each client keeps a random subset of ≤ cap calibration scores
        crng = np.random.RandomState(seed + 77)
        for cap in calib_caps:
            sub = [s if len(s) <= cap else crng.choice(s, cap, replace=False) for s in s_cal]
            methods[f"conformal_local_cap{cap}_a{a}"] = [th.conformal(x, a) for x in sub]
    for name, rhos in methods.items():
        a = float(name.split("_a")[-1])
        fpr_id = [float(np.mean(s < r)) if len(s) else np.nan for s, r in zip(s_id, rhos)]
        fpr_te = [float(np.mean(s < r)) if len(s) else np.nan for s, r in zip(s_teb, rhos)]
        tpr_te = [float(np.mean(s < r)) if len(s) else np.nan for s, r in zip(s_tea, rhos)]
        cap = int(name.split("_cap")[1].split("_")[0]) if "_cap" in name else None
        ns = np.array([min(len(s), cap) if cap else len(s) for s in s_cal])
        ok = ~np.isnan(fpr_te)
        res[name] = {
            "alpha": a,
            "id": M.fpr_violation(np.array(fpr_id)[ok], ns[ok], a),
            "test": M.fpr_violation(np.array(fpr_te)[ok], ns[ok], a),
            "tpr_mean": float(np.nanmean(tpr_te)),
            "trivial_frac": float(np.mean(np.isinf(rhos))),
            "fpr_test_per_client": fpr_te, "fpr_id_per_client": fpr_id, "tpr_per_client": tpr_te,
        }
    # pooled ranking metrics (threshold-free) on the global test set
    te = D["test"]
    s_te = sf(embed(model, te["X"], normalize=True))
    res["ranking_test"] = M.ranking(te["y"], s_te)
    res["ranking_val"] = M.ranking(D["val"]["y"], sf(embed(model, D["val"]["X"], normalize=True)))
    return res
