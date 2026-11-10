"""Unified dataset builder. Returns dict of splits, each a dict of numpy arrays:
X (float32), y (0/1), cat (attack category / 'Normal'), group (benign heterogeneity key: NSL-KDD service group,
N-BaIoT device, CICIoT 'all'), novel (test only: attack type unseen in training).
Splits: train, val (tuning; benign+attacks), calib (benign only, disjoint), test.
"""
import json
import zlib
import os

import numpy as np
import pandas as pd

from . import nslkdd as nk
from . import paths

CACHE = os.path.join(paths.PROCESSED, "_cache")


def _service_group(df, top=9):
    # benign heterogeneity key: the 9 most frequent services in KDDTrain+ benign, else 'other'
    top_s = ["http", "private", "domain_u", "smtp", "ftp_data", "eco_i", "other", "ecr_i", "telnet"][:top]
    return np.where(df.service.isin(top_s), df.service, "other_svc")


def nslkdd(seed=0, fs="paper", scaler="minmax", log1p=False, train_on="split", drop=None, return_frames=False,
           holdout=None, n_anova=3, test_file="full"):
    """fs: 'paper' (FeCo drop list), 'procedure' (FS stats on the train split), 'none', or explicit `drop` list
    (e.g. from federated FS). train_on='full' trains on the whole KDDTrain+ (FeCo protocol) — val then overlaps train
    and must not be used for selection."""
    tr, te = nk.load_raw(test_file)
    sp = np.load(os.path.join(nk.OUT, f"splits_seed{seed}.npz"))
    i_tr = np.arange(len(tr)) if train_on == "full" else sp["train"]
    i_vt, i_va = sp["val_tune"], sp["val"]
    zd = None
    if holdout is not None:  # leave-one-category-out: held-out rows of KDDTrain+ form the zero-day split
        cat = tr.category.values
        held = np.r_[i_tr[cat[i_tr] == holdout], i_va[cat[i_va] == holdout]]
        i_tr, i_vt, i_va = i_tr[cat[i_tr] != holdout], i_vt[cat[i_vt] != holdout], i_va[cat[i_va] != holdout]
        zd = np.sort(np.r_[held, i_vt[tr.y.values[i_vt] == 0]])
    dtr = tr.iloc[i_tr]
    if drop is None:
        if fs == "paper":
            drop = nk.FECO_PAPER_DROP
        elif fs == "procedure":
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                drop, _ = nk.select_features(nk.local_stats(dtr, dtr.category.values), n_anova=n_anova)
        elif fs == "none":
            drop = [c for c in nk.COLS[:41] if dtr[c].nunique() <= 1 and c not in nk.CATEGORICAL]
        else:
            raise ValueError(fs)
    enc = nk.Encoder(drop, scaler=scaler, log1p=log1p).fit(dtr)

    def pack(df, novel=None):
        d = {"X": enc.transform(df), "y": df.y.values.astype(np.int64), "cat": df.category.values.astype(str),
             "label": df.label.values.astype(str), "group": _service_group(df)}
        if novel is not None:
            d["novel"] = novel
        return d

    train_labels = set(dtr.label)
    has_sub = te.attrs.get("has_subclass", False)
    novel = (te.label.isin(nk.TEST_ONLY).values & (te.y.values == 1)) if has_sub else None
    out = {"name": "nslkdd", "dim": enc.dim, "drop": list(drop),
           "train": pack(dtr), "val": pack(tr.iloc[i_vt]), "val_all": pack(tr.iloc[i_va]),
           "calib": pack(tr.iloc[sp["calib"]]), "test": pack(te, novel), "test_has_subclass": has_sub,
           "train_labels": sorted(train_labels), "holdout": holdout}
    if zd is not None:
        out["zeroday"] = pack(tr.iloc[zd], novel=(tr.category.values[zd] == holdout))
    if return_frames:
        out["frames"] = (tr, te, sp)
    return out


def _nbaiot_split_indices(df, dev, seed, cap_train_att, cap_test_att):
    rng = np.random.RandomState(seed * 7919 + zlib.crc32(dev.encode()) % 1000)
    fam = df.family.astype(str).values
    att = df.attack.astype(str).values
    ib = np.flatnonzero(fam == "benign")  # file order = time order
    n = len(ib)
    a, c = int(0.6 * n), int(0.8 * n)
    # time order: train (0-60%) | calibration window (60-80%) | test (80-100%). The window is split at random 50/50
    # into val / calib so the in-distribution hold-out is exchangeable with calib.
    win = rng.permutation(ib[a:c])
    ben = {"train": ib[:a], "val": np.sort(win[: len(win) // 2]), "calib": np.sort(win[len(win) // 2:]), "test": ib[c:]}
    mir = sorted(set(att[fam == "mirai"]))
    gaf = sorted(set(att[fam == "gafgyt"]))
    if mir:
        tr_att = list(rng.choice(mir, 3, replace=False))
    else:
        tr_att = list(rng.choice(gaf, 3, replace=False))
    te_att = [x for x in mir + gaf if x not in tr_att]
    tr_idx, va_idx, te_idx = [], [], []
    for x in tr_att:
        ix = rng.permutation(np.flatnonzero(att == x))[:cap_train_att]
        k = int(0.1 * len(ix))
        va_idx.append(ix[:k])
        tr_idx.append(ix[k:])
    for x in te_att:
        te_idx.append(rng.permutation(np.flatnonzero(att == x))[:cap_test_att])
    return ben, np.concatenate(tr_idx), np.concatenate(va_idx), np.concatenate(te_idx), tr_att, te_att


def nbaiot(seed=0, devices=None, cap_train_att=6000, cap_test_att=3000, scaler="minmax", log1p=True,
           corr_thr=0.98, benign_frac=1.0):
    """Per-device: benign time-ordered 60/10/10/20; attacks FeCo protocol (3/5 Mirai train, rest + BASHLITE test;
    no-Mirai devices: 3/5 BASHLITE train). 10% of training attacks → val. Attack rows are capped per attack type
    (compute cut, logged); benign calib/test kept complete. Features: zero-variance and |Spearman|>corr_thr removal
    on a train subsample; log1p (heavy-tailed Kitsune stats) then min-max fit on train."""
    from . import nbaiot as nb
    devices = devices or nb.DEVICES
    splits = {k: [] for k in ("train", "val", "calib", "test")}
    meta = {}
    for dev in devices:
        df = nb.load_device(dev)
        ben, itr, iva, ite, tr_att, te_att = _nbaiot_split_indices(df, dev, seed, cap_train_att, cap_test_att)
        if benign_frac < 1.0:
            ben["train"] = ben["train"][: int(benign_frac * len(ben["train"]))]
        meta[dev] = {"train_attacks": tr_att, "test_attacks": te_att, **{k: len(v) for k, v in ben.items()}}
        X = df[[f"f{i}" for i in range(115)]].values
        att = df.attack.astype(str).values
        for name, idx in (("train", np.r_[ben["train"], itr]), ("val", np.r_[ben["val"], iva]),
                          ("calib", ben["calib"]), ("test", np.r_[ben["test"], ite])):
            y = (att[idx] != "benign").astype(np.int64)
            d = {"X": X[idx], "y": y, "cat": np.where(y == 1, att[idx], "Normal"), "group": np.array([dev] * len(idx)),
                 "label": att[idx]}
            if name == "test":
                d["novel"] = np.isin(att[idx], te_att)
            splits[name].append(d)
        del df, X
    S = {k: {f: np.concatenate([d[f] for d in v]) for f in v[0]} for k, v in splits.items()}
    Xtr = S["train"]["X"].astype(np.float64)
    if log1p:
        tf = lambda A: np.sign(A) * np.log1p(np.abs(A))  # noqa
    else:
        tf = lambda A: A  # noqa
    Ttr = tf(Xtr)
    var = Ttr.var(0)
    keep = np.flatnonzero(var > 1e-12)
    rng = np.random.RandomState(seed)
    sub = Ttr[rng.choice(len(Ttr), min(50000, len(Ttr)), replace=False)][:, keep]
    from scipy.stats import spearmanr
    R = np.abs(np.nan_to_num(spearmanr(sub).statistic))
    removed = []
    alive = list(range(len(keep)))
    for i in range(len(keep)):
        if i not in alive:
            continue
        for j in range(i + 1, len(keep)):
            if j in alive and R[i, j] > corr_thr:
                alive.remove(j)
                removed.append(int(keep[j]))
    keep = keep[alive]
    lo, hi = Ttr[:, keep].min(0), Ttr[:, keep].max(0)
    rngs = np.where(hi - lo > 0, hi - lo, 1.0)
    for k in S:
        S[k]["X"] = ((tf(S[k]["X"].astype(np.float64))[:, keep] - lo) / rngs).astype(np.float32)
    names = json.load(open(os.path.join(nb.OUT, "feature_names.json")))
    return {"name": "nbaiot", "dim": len(keep), "train": S["train"], "val": S["val"], "calib": S["calib"],
            "test": S["test"], "meta": meta, "test_has_subclass": True,
            "features_kept": [names[i] for i in keep], "features_removed_zero_var": [names[i] for i in np.flatnonzero(var <= 1e-12)],
            "features_removed_corr": [names[i] for i in removed]}


def _fit_transform_tabular(S, seed, log1p=True, corr_thr=0.98, names=None):
    tf = (lambda A: np.sign(A) * np.log1p(np.abs(A))) if log1p else (lambda A: A)  # noqa
    Ttr = tf(S["train"]["X"].astype(np.float64))
    var = Ttr.var(0)
    keep = np.flatnonzero(var > 1e-12)
    rng = np.random.RandomState(seed)
    sub = Ttr[rng.choice(len(Ttr), min(50000, len(Ttr)), replace=False)][:, keep]
    from scipy.stats import spearmanr
    R = np.abs(np.nan_to_num(spearmanr(sub).statistic))
    alive, removed = list(range(len(keep))), []
    for i in range(len(keep)):
        if i not in alive:
            continue
        for j in range(i + 1, len(keep)):
            if j in alive and R[i, j] > corr_thr:
                alive.remove(j)
                removed.append(int(keep[j]))
    zv = [int(i) for i in np.flatnonzero(var <= 1e-12)]
    keep = keep[alive]
    lo, hi = Ttr[:, keep].min(0), Ttr[:, keep].max(0)
    rngs = np.where(hi - lo > 0, hi - lo, 1.0)
    for k in S:
        S[k]["X"] = ((tf(S[k]["X"].astype(np.float64))[:, keep] - lo) / rngs).astype(np.float32)
    nm = (lambda ix: [names[i] for i in ix]) if names else (lambda ix: list(ix))  # noqa
    return S, {"dim": len(keep), "features_kept": nm(keep), "features_removed_zero_var": nm(zv),
               "features_removed_corr": nm(removed)}


CICIOT_SPLIT_VERSION = "v2_file_block"  # v1 = row-level random split within class (leakage risk: adjacent flows in train and test)
SPLITS = ("train", "val", "calib", "test")


def _cap(ix, cap, rng):
    if cap is None or len(ix) <= cap:
        return np.sort(ix)
    return np.sort(rng.choice(ix, cap, replace=False))


def _ciciot_split_indices(cls, fil, rif, seed, cap_train_benign=100000, cap_train_att=10000, cap_val_att=2000,
                          cap_test_att=10000, cap_val_benign=20000, benign_cls="Benign_Final"):
    """Leakage-aware CICIoT2023 split. Rows of one CSV come from one pcap capture; neighbouring rows are
    temporally adjacent flows, so a row-level random split puts near-duplicates into train and test.
    - A class with ≥ len(splits) source files (Benign_Final: 4 files → train/val/calib/test) is split BY FILE: a
      seed-dependent permutation of its files, one file per split (extra files → train).
    - Otherwise (every attack class has 1–2 files; attacks need train/val/test): contiguous row blocks within each file
      in file order: [0,60%) train | [60,70%) val | [70,80%) unused gap | [80,100%) test (benign fallback: gap = calib).
    Caps (compute cut) are random subsets within a split; benign calib/test are never capped.
    Returns ({split: sorted row indices}, info)."""
    rng = np.random.RandomState(seed)
    idx = {k: [] for k in SPLITS}
    info = {"version": CICIOT_SPLIT_VERSION, "file_split": {}, "block_split": [], "rows": {}}
    for c in sorted(set(cls)):
        ic = np.flatnonzero(cls == c)
        benign = c == benign_cls
        names = list(SPLITS) if benign else ["train", "val", "test"]
        files = sorted(set(fil[ic]))
        part = {k: [] for k in SPLITS}
        if len(files) >= len(names):
            perm = [files[i] for i in rng.permutation(len(files))]
            assign = {f: names[i] if i < len(names) else "train" for i, f in enumerate(perm)}
            for f in files:
                part[assign[f]].append(ic[fil[ic] == f])
            info["file_split"][c] = {k: sorted(f for f, v in assign.items() if v == k) for k in names}
        else:
            for f in files:
                jf = ic[fil[ic] == f]
                jf = jf[np.argsort(rif[jf], kind="stable")]
                n = len(jf)
                a, b, c3 = int(0.6 * n), int(0.7 * n), int(0.8 * n)
                part["train"].append(jf[:a])
                part["val"].append(jf[a:b])
                if benign:
                    part["calib"].append(jf[b:c3])
                part["test"].append(jf[c3:])
            info["block_split"].append(c)
        part = {k: (np.concatenate(v) if v else np.array([], np.int64)) for k, v in part.items()}
        caps = ({"train": cap_train_benign, "val": cap_val_benign, "calib": None, "test": None} if benign else
                {"train": cap_train_att, "val": cap_val_att, "calib": 0, "test": cap_test_att})
        for k in SPLITS:
            if caps[k] == 0:
                continue
            sel = _cap(part[k], caps[k], rng)
            idx[k].append(sel)
            info["rows"].setdefault(c, {})[k] = int(len(sel))
    return {k: np.sort(np.concatenate(v)) for k, v in idx.items()}, info


def _ciciot_base(seed, cap_train_benign, cap_train_att, cap_val_att, cap_test_att, cap_val_benign):
    """Raw (untransformed) split rows, cached per seed+caps+split version so workers never load the 4.3M-row parquet."""
    import pickle
    key = (f"ciciot_base_{CICIOT_SPLIT_VERSION}_s{seed}_{cap_train_benign}_{cap_train_att}_{cap_val_att}_"
           f"{cap_test_att}_{cap_val_benign}.pkl")
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    from . import ciciot as cc
    df = cc.load()
    feats = [c for c in df.columns if c not in ("cls", "family", "file", "row_in_file")]
    X = df[feats].values
    ok = np.isfinite(X).all(1)
    df, X = df[ok].reset_index(drop=True), X[ok]
    cls = df.cls.astype(str).values
    fam = df.family.astype(str).values
    fil = df.file.astype(str).values
    rif = df.row_in_file.values
    idx, info = _ciciot_split_indices(cls, fil, rif, seed, cap_train_benign, cap_train_att, cap_val_att, cap_test_att,
                                      cap_val_benign)
    S = {}
    for k, ii in idx.items():
        y = (fam[ii] != "Benign").astype(np.int64)
        # benign files are split-disjoint, so the file can no longer be a heterogeneity key shared by
        # train/calib/test: one benign group
        S[k] = {"X": X[ii].astype(np.float32), "y": y, "cat": np.where(y == 1, fam[ii], "Normal"), "label": cls[ii],
                "group": np.where(y == 0, "benign", "attack")}
    base = {"S": S, "feats": feats, "n_dropped_nonfinite": int((~ok).sum()), "split_info": info}
    os.makedirs(CACHE, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    pickle.dump(base, open(tmp, "wb"), protocol=4)
    os.replace(tmp, path)
    return base


def ciciot(seed=0, cap_train_benign=100000, cap_train_att=10000, cap_val_att=2000, cap_test_att=10000,
           cap_val_benign=20000, log1p=True, corr_thr=0.98, holdout=None):
    """Split v2 (see _ciciot_split_indices): benign by capture file (one file per split), attacks by contiguous
    row blocks 60/10/(10 gap)/20 within each file; calib keeps benign only. Non-finite rows dropped.
    Caps (compute cut, logged): train benign ≤100k, per-class attacks train ≤10k / val ≤2k / test ≤10k, val benign ≤20k.
    Benign calib and test kept complete. group = 'benign' (single group). holdout: family removed from
    train/val (zero-day: appears only in test)."""
    base = _ciciot_base(seed, cap_train_benign, cap_train_att, cap_val_att, cap_test_att, cap_val_benign)
    S = {k: {f: v.copy() for f, v in d.items()} for k, d in base["S"].items()}
    if holdout is not None:
        for k in ("train", "val"):
            keep = S[k]["cat"] != holdout
            S[k] = {f: v[keep] for f, v in S[k].items()}
    train_fams = set(S["train"]["cat"][S["train"]["y"] == 1])
    S["test"]["novel"] = (S["test"]["y"] == 1) & ~np.isin(S["test"]["cat"], list(train_fams))
    S, info = _fit_transform_tabular(S, seed, log1p, corr_thr, names=base["feats"])
    return {"name": "ciciot", **info, **S, "test_has_subclass": True, "holdout": holdout,
            "n_dropped_nonfinite": base["n_dropped_nonfinite"], "split_info": base.get("split_info"),
            "counts": {k: {"benign": int((v["y"] == 0).sum()), "attack": int((v["y"] == 1).sum())} for k, v in S.items()}}
