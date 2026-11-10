"""Unit tests for ROC metrics, losses, robust quantile, Byzantine attacks, client assignment, k-means / vMF,
feature selection, test-file handling and split disjointness.
Real-data tests read only small files / metadata columns and finish in seconds."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from src.attacks.byzantine import Byzantine
from src.calib import thresholds as th
from src.calib.robust import robust_quantile
from src.data import nslkdd as nk
from src.data.datasets import _ciciot_split_indices, _nbaiot_split_indices
from src.data.paths import PROCESSED
from src.eval.metrics import ranking, tpr_at_fpr, fpr_at_tpr
from src.fl.fedeval import assign, share_matrix
from src.models import detect as det
from src.models.encoder import MLP, embed
from src.models.losses import clad_loss, feco_loss, svdd_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def test_tpr_at_fpr_tied_scores_no_point_dropping():
    # every score level holds one benign + one attack → ROC points (0,0),(.25,.25),...,(1,1) are collinear;
    # sklearn 1.7's drop_intermediate=True keeps (0,0),(.25,.25),(1,1) and reports TPR=0.25 at FPR≤0.5 (truth 0.5).
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    anom = np.array([4, 4, 3, 3, 2, 2, 1, 1], float)
    assert tpr_at_fpr(y, anom, 0.5) == 0.5
    assert tpr_at_fpr(y, anom, 0.5, drop_intermediate=True) == 0.25  # documents the drop_intermediate bias
    assert fpr_at_tpr(y, anom, 0.5) == 0.5
    r = ranking(y, -anom)
    assert r["tpr@1"] == 0.0 and r["tpr@5"] == 0.0 and abs(r["auroc"] - 0.5) < 1e-12
    # continuous scores: both versions agree
    rng = np.random.RandomState(0)
    y2 = np.r_[np.zeros(500), np.ones(500)].astype(int)
    a2 = np.r_[rng.randn(500), rng.randn(500) + 1.5]
    for t in (0.01, 0.05, 0.2):
        assert tpr_at_fpr(y2, a2, t) == tpr_at_fpr(y2, a2, t, drop_intermediate=True)


def test_losses_nb_slicing_matches_masks():
    torch.manual_seed(0)
    for nb, na in ((7, 5), (12, 0), (3, 9)):
        z = torch.randn(nb + na, 16, dtype=torch.float64)
        y = torch.tensor([0] * nb + [1] * na)
        for tau in (0.1, 0.5):
            a, b = feco_loss(z, y, tau), feco_loss(z, y, tau, nb=nb)
            assert torch.allclose(a, b, rtol=1e-12, atol=1e-12)
        assert torch.allclose(clad_loss(z, y), clad_loss(z, y, nb=nb), rtol=1e-12, atol=1e-12)
        c = torch.randn(16, dtype=torch.float64)
        assert torch.allclose(svdd_loss(z, y, c), svdd_loss(z, y, c, nb=nb), rtol=1e-12, atol=1e-12)


def test_robust_quantile_excludes_poisoned_clients():
    rng = np.random.RandomState(0)
    honest = [np.tanh(rng.randn(400) * 0.3 + 0.5) for _ in range(14)]
    byz = [np.full(1200, -1.0) for _ in range(6)]  # calibration poisoning: mass at -1, counts x3
    hists = [th.histogram(s) for s in honest + byz]
    a = 0.05
    rho_naive = th.from_histogram(sum(hists), a)
    rho_rob, keep, M = robust_quantile(hists, a, f_max=0.3)
    rho_honest = th.from_histogram(sum(hists[:14]), a)
    assert rho_naive == -1.0  # poisoned global quantile collapses
    assert set(keep.tolist()) <= set(range(14)) and len(keep) == 20 - 6  # K_b = K − ⌈0.3K⌉ = 14
    assert abs(rho_rob - rho_honest) < 1e-9
    assert M[14:].min() > M[:14].max()
    # clean federation: robust quantile still drops ⌈f_max·K⌉ honest clients
    _, keep2, _ = robust_quantile([th.histogram(s) for s in honest], a, f_max=0.3)
    assert len(keep2) == 14 - int(np.ceil(0.3 * 14))


def _toy_fed(seed=0, n_clients=10):
    rng = np.random.RandomState(seed)
    nb, na, d = 60, 30, 8
    X, y, parts = [], [], []
    off = 0
    for c in range(n_clients):
        xb = rng.randn(nb, d) * 0.3 + 1.0
        xa = rng.randn(na, d) * 0.3 - 1.0
        X += [xb, xa]
        y += [0] * nb + [1] * na
        parts.append(np.arange(off, off + nb + na))
        off += nb + na
    X = np.concatenate(X).astype(np.float32)
    y = np.array(y)
    D = {"train": {"X": X, "y": y}, "calib": {"X": X[y == 0][:200], "y": np.zeros(200, int)}}
    C = {"train": parts, "calib_benign": [np.arange(i * 20, (i + 1) * 20) for i in range(n_clients)]}
    return D, C


def test_byzantine_ids_and_calibration_poisoning():
    D, C = _toy_fed()
    b = Byzantine({"attack": "calib", "frac": 0.3, "boost": 3.0, "calib_value": -1.0}, D, C, seed=0)
    assert len(b.ids) == 3
    s = np.linspace(0.2, 0.9, 20)
    for cid in range(10):
        out = b.calib_poison(cid, s)
        if cid in b.ids:
            assert len(out) == 60 and np.all(out == -1.0)
        else:
            assert out is s
    rep = [b.calib_poison(i, s) for i in range(10)]
    assert th.from_histogram(sum(th.histogram(r) for r in rep), 0.05) == -1.0
    assert robust_quantile([th.histogram(r) for r in rep], 0.05, f_max=0.3)[0] > 0


def test_byzantine_prototype_poisoning():
    D, C = _toy_fed()
    torch.manual_seed(0)
    model = MLP(8, hidden=(16, 16), out=4)
    for K in (1, 2):
        b = Byzantine({"attack": "proto", "frac": 0.2, "boost": 3.0}, D, C, seed=1)
        b.set_model(model)
        Zb = embed(model, D["train"]["X"][D["train"]["y"] == 0][:60])
        Ch, ch = det.unit(Zb.mean(0, keepdims=True)), np.array([60.0])
        cid = sorted(b.ids)[0]
        honest = [i for i in range(10) if i not in b.ids][0]
        Cf, cf = b.proto_poison(cid, np.repeat(Ch, K, 0), np.full(K, 60.0 / K))
        Ca = embed(model, D["train"]["X"][b.att_rows[cid]])
        assert abs(cf.sum() - 3.0 * 60.0) < 1e-6  # counts inflated by boost × own benign count
        # fake prototypes sit in the attacker's attack-embedding region
        assert np.max(Cf @ det.unit(Ca.mean(0))) > 0.9
        C2, c2 = b.proto_poison(honest, Ch, ch)
        assert C2 is Ch and c2 is ch


def test_byzantine_whitelisting_pulls_attacks_and_boosts():
    D, C = _toy_fed()
    torch.manual_seed(0)
    model = MLP(8, hidden=(16, 16), out=4)
    b = Byzantine({"attack": "whitelist", "frac": 0.2, "boost": 5.0, "atk_lr_mult": 10.0}, D, C, seed=2)
    cid = sorted(b.ids)[0]
    p = C["train"][cid]
    Xc, yc = torch.as_tensor(D["train"]["X"][p]), D["train"]["y"][p]

    def gap(m):
        Z = embed(m, Xc.numpy())
        vb = det.unit(Z[yc == 0].mean(0))
        return float((Z[yc == 1] @ vb).mean())
    before = gap(model)
    import copy
    m2 = b.local(cid, copy.deepcopy(model), Xc, yc, {"epochs": 4, "lr": 1e-3, "momentum": 0.9},
                 np.random.RandomState(0), model)
    assert gap(m2) > before + 0.05  # attack embeddings pulled toward the benign mean
    g, v = torch.zeros(5), torch.ones(5)
    assert torch.allclose(b.post(cid, v, g), 5 * torch.ones(5))
    b1 = Byzantine({"attack": "whitelist", "frac": 0.2, "boost": 1.0}, D, C, seed=2)
    assert b1.post(cid, v, g) is v


def test_share_matrix_and_assign():
    rng = np.random.RandomState(0)
    G = ["a", "b", "c"]
    P, G2 = share_matrix(G, 9, "natural", None, rng)
    assert G2 == G and np.allclose(P.sum(0), 1) and (P > 0).sum() == 9
    assert all((P[i] > 0).sum() == 1 for i in range(9))
    for sch in ("dirichlet", "iid"):
        P, _ = share_matrix(G, 20, sch, 0.5, rng)
        assert P.shape == (20, 3) and np.allclose(P.sum(0), 1) and (P >= 0).all()
    groups = np.array(rng.choice(G, 3000))
    idx = np.arange(3000)
    P, _ = share_matrix(G, 20, "dirichlet", 1.0, rng)
    parts = assign(groups, idx, P, G, rng)
    allp = np.concatenate(parts)
    assert len(allp) == len(np.unique(allp)) == 3000  # disjoint and complete
    for gi, g in enumerate(G):
        ng = (groups == g).sum()
        got = np.array([(groups[p] == g).sum() for p in parts])
        assert np.abs(got - P[:, gi] * ng).max() <= 1.0 + 1e-9  # rounding of cumulative cut points


def _sphere_clusters(rng, centers, n, kappa_noise):
    X = np.concatenate([c + rng.randn(n, len(c)) * s for c, s in zip(centers, kappa_noise)])
    return det.unit(X).astype(np.float32)


@pytest.mark.parametrize("impl", ["fast", "legacy"])
def test_spherical_kmeans_recovers_clusters(impl):
    det.KMEANS_IMPL = impl
    try:
        rng = np.random.RandomState(0)
        centers = det.unit(rng.randn(3, 16))
        Z = _sphere_clusters(rng, centers, 300, [0.05, 0.05, 0.05])
        C, cnt = det.spherical_kmeans(Z, 3, seed=0)
        assert np.allclose(np.linalg.norm(C, axis=1), 1, atol=1e-5)
        assert np.min(np.max(C @ centers.T, axis=0)) > 0.99
        assert sorted(cnt.astype(int)) == [300, 300, 300]
    finally:
        det.KMEANS_IMPL = "fast"


def test_kmeans_fast_equals_legacy_assignments():
    rng = np.random.RandomState(1)
    Z = det.unit(rng.randn(2000, 32)).astype(np.float32)
    out = {}
    for impl in ("fast", "legacy"):
        det.KMEANS_IMPL = impl
        out[impl] = det.spherical_kmeans(Z, 8, seed=3)
    det.KMEANS_IMPL = "fast"
    assert np.abs(out["fast"][0] - out["legacy"][0]).max() < 1e-4
    assert np.array_equal(out["fast"][1], out["legacy"][1])
    a = np.array([0, 2, 2, 1])
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    assert np.allclose(det.scatter_sum(a, X, 3), [[0, 1, 2], [9, 10, 11], [9, 11, 13]])


def test_vmf_mixture_fit_and_score():
    rng = np.random.RandomState(0)
    centers = det.unit(rng.randn(2, 16))
    Z = _sphere_clusters(rng, centers, 500, [0.05, 0.3])  # tight and loose cluster
    m = det.fit_vmf_mixture(Z, 2, seed=0)
    order = np.argmax(m["mu"] @ centers.T, axis=0)
    assert np.all(np.max(m["mu"] @ centers.T, axis=0) > 0.98)
    assert m["kappa"][order[0]] > m["kappa"][order[1]]  # tighter cluster → larger κ
    assert abs(m["pi"].sum() - 1) < 1e-9
    far = det.unit(-centers.sum(0, keepdims=True)).astype(np.float32)
    assert det.score_vmf(Z[:10], m).min() > det.score_vmf(far, m)[0]
    assert np.isfinite(det.log_cd(np.array([1e-3, 10.0, 1e4]), 128)).all()


def test_select_features_rules_and_aggregation():
    num = ["a", "b", "c", "d", "e", "f"]
    st = {"num": num, "var": np.array([1, 1, 0, 1, 1, 1.0]),
          "rho": np.eye(6), "F": np.array([5.0, 1.0, 9.0, 8.0, 0.5, 7.0]), "mi": np.zeros(8)}
    st["rho"][0, 1] = st["rho"][1, 0] = 0.99  # a~b: drop lower-F member (b)
    st["rho"][3, 5] = st["rho"][5, 3] = 0.95  # d~f: drop f (F 7 < 8)
    st["rho"][0, 3] = st["rho"][3, 0] = 0.90  # a~d: drop a (F 5 < 8)
    st["mi"][0] = 0.0
    st["mi"][1:] = 0.2
    drop, info = nk.select_features(st, n_corr=3, n_anova=1, mi_eps=5e-3)
    assert info["zero_var"] == ["c"]
    assert [x["dropped"] for x in info["spearman"]] == ["b", "f", "a"]
    assert info["mi_dropped"] == [nk.MI_FEATURES[0]]
    assert info["anova_dropped"] == ["e"]
    s1 = {k: (np.asarray(v) * 1.0 if k not in ("num", "n") else v) for k, v in st.items()}
    s1["n"] = 30
    s2 = {**s1, "F": s1["F"] * 3, "n": 10}
    agg = nk.aggregate_stats([s1, s2])
    assert agg["n"] == 40 and np.allclose(agg["F"], 0.75 * s1["F"] + 0.25 * s2["F"])


def test_feco_feature_selection_reproduces_paper_drop_list():
    tr = nk._read(os.path.join(nk.RAW, "KDDTrain+.txt"))
    tr["category"] = tr.label.map(nk.CATEGORY)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        drop, _ = nk.select_features(nk.local_stats(tr, tr.category.values))
    assert sorted(drop) == sorted(nk.FECO_PAPER_DROP)
    assert nk.Encoder(drop).fit(tr).dim == 112


def test_nslkdd_full_test_file_is_official():
    tr, te = nk.load_raw()
    assert len(te) == 22544 and te.attrs["has_subclass"]
    assert not (te.service == "Attack4").any()
    assert set(te.label) <= set(nk.CATEGORY)
    assert (te.y == 1).sum() == 22544 - 9711
    assert set(te.service) <= set(tr.service)  # no all-zero one-hot rows


def test_cache_key_includes_test_file_md5():
    from src.pipeline import cache_key
    d = {"name": "nslkdd", "fs": "procedure"}
    k_full, v_full = cache_key(d, 0)
    k_mir, v_mir = cache_key({**d, "test_file": "mirror"}, 0)
    assert v_full["test_md5"] == nk.file_md5(nk.TEST_FULL) and v_full["test_md5"] in k_full
    assert v_mir["test_md5"] == nk.file_md5(nk.TEST_MIRROR) and k_full != k_mir
    k_c, v_c = cache_key({"name": "ciciot"}, 0)
    from src.data.datasets import CICIOT_SPLIT_VERSION
    assert v_c == {"split": CICIOT_SPLIT_VERSION} and CICIOT_SPLIT_VERSION in k_c
    k_n, v_n = cache_key({"name": "nbaiot"}, 0)  # unchanged → existing caches stay valid
    assert v_n is None and k_n == json.dumps({"name": "nbaiot", "seed": 0}, sort_keys=True)


def test_train_on_full_marks_conformal_invalid():
    from src.pipeline import evaluate_scores
    rng = np.random.RandomState(0)
    D = {"calib": {"X": rng.randn(50, 2)}, "val": {"X": rng.randn(40, 2), "y": np.r_[np.zeros(20), np.ones(20)].astype(int),
                                                     "cat": np.array(["Normal"] * 20 + ["DoS"] * 20)}}
    sc = {"template": lambda X: X[:, 0]}
    st = {"template": rng.randn(100)}
    for train_on, pre in (("full", "invalid_conf"), ("split", "conf")):
        r = evaluate_scores(D, {"data": {"train_on": train_on}, "thresholds": {"alphas": [0.05]}}, sc, st, ("val",))
        assert f"{pre}_a0.05" in r["template"]["rho"] and f"{pre}_a0.05" in r["template"]["val"]
        other = "conf" if pre == "invalid_conf" else "invalid_conf"
        assert f"{other}_a0.05" not in r["template"]["rho"]


@pytest.mark.parametrize("seed", range(5))
def test_nslkdd_splits_disjoint(seed):
    sp = np.load(os.path.join(nk.OUT, f"splits_seed{seed}.npz"))
    tr = nk._read(os.path.join(nk.RAW, "KDDTrain+.txt"))
    y = (tr.label != "normal").values
    assert len(np.intersect1d(sp["train"], sp["val"])) == 0
    assert len(np.intersect1d(sp["val_tune"], sp["calib"])) == 0
    assert np.array_equal(np.sort(np.r_[sp["val_tune"], sp["calib"]]), np.sort(sp["val"]))
    assert not y[sp["calib"]].any()
    assert len(sp["train"]) + len(sp["val"]) == len(tr)


def test_nbaiot_split_indices_disjoint_time_ordered():
    rng = np.random.RandomState(0)
    fam = ["benign"] * 1000 + ["mirai"] * 500 + ["gafgyt"] * 500
    att = ["benign"] * 1000 + list(np.repeat([f"mirai_m{i}" for i in range(5)], 100)) + \
        list(np.repeat([f"gafgyt_g{i}" for i in range(5)], 100))
    df = pd.DataFrame({"family": fam, "attack": att, "x": rng.randn(2000)})
    for seed in range(3):
        ben, itr, iva, ite, tr_att, te_att = _nbaiot_split_indices(df, "dev", seed, 60, 50)
        allb = np.concatenate(list(ben.values()))
        assert len(allb) == len(np.unique(allb)) == 1000
        assert ben["train"].max() < min(ben["val"].min(), ben["calib"].min())
        assert max(ben["val"].max(), ben["calib"].max()) < ben["test"].min()
        assert len(np.intersect1d(itr, iva)) == 0 and len(np.intersect1d(itr, ite)) == 0
        assert set(tr_att).isdisjoint(te_att) and len(tr_att) == 3 and all(a.startswith("mirai") for a in tr_att)
        assert set(df.attack.values[ite]) == set(te_att)
    df2 = df[df.family != "mirai"].reset_index(drop=True)  # no-Mirai device: 3/5 BASHLITE train
    _, _, _, _, tr_att, te_att = _nbaiot_split_indices(df2, "dev2", 0, 60, 50)
    assert len(tr_att) == 3 and len(te_att) == 2 and all(a.startswith("gafgyt") for a in tr_att + te_att)


def test_ciciot_split_by_file_and_blocks():
    meta = pd.read_parquet(os.path.join(PROCESSED, "ciciot2023", "all.parquet"), columns=["cls", "file", "row_in_file"])
    cls, fil, rif = meta.cls.astype(str).values, meta.file.astype(str).values, meta.row_in_file.values
    for seed in (0, 1):
        idx, info = _ciciot_split_indices(cls, fil, rif, seed)
        ks = list(idx)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                assert len(np.intersect1d(idx[ks[i]], idx[ks[j]])) == 0, (ks[i], ks[j])
        assert set(cls[idx["calib"]]) == {"Benign_Final"}
        # benign: each split uses its own capture file
        bfiles = {k: set(fil[idx[k]][cls[idx[k]] == "Benign_Final"]) for k in ks}
        assert all(len(v) == 1 for v in bfiles.values())
        assert len(set.union(*bfiles.values())) == 4
        # attacks: contiguous blocks per file, train < val < test in file order
        for c in info["block_split"]:
            for f in set(fil[cls == c]):
                r = {k: rif[idx[k]][(cls[idx[k]] == c) & (fil[idx[k]] == f)] for k in ("train", "val", "test")}
                assert r["train"].max() < r["val"].min() and r["val"].max() < r["test"].min()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_embed_cuda_returns_cpu_numpy_matching_cpu():
    torch.manual_seed(0)
    m = MLP(10)
    X = np.random.RandomState(0).randn(300, 10).astype(np.float32)
    a = embed(m, X)
    b = embed(m.to("cuda"), X)
    assert isinstance(b, np.ndarray) and b.shape == a.shape and np.abs(a - b).max() < 1e-4
