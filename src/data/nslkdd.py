"""NSL-KDD loading, FeCo category mapping (FeCo TDSC Table II), FeCo two-step feature selection, encoding."""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import f_classif, mutual_info_classif

from . import paths

RAW = os.path.join(paths.RAW, "nslkdd")
OUT = os.path.join(paths.PROCESSED, "nslkdd")

COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "label", "difficulty"]
CATEGORICAL = ["protocol_type", "service", "flag"]
# FeCo Fig. 4(c): "categorical" importance (MI) is computed for these 8 (indices 32-39 in FeCo's legend)
BINARY = ["land", "is_host_login", "is_guest_login", "logged_in", "root_shell"]
MI_FEATURES = BINARY + CATEGORICAL

# FeCo Table II (TDSC) / Table I (INFOCOM). FeCo puts httptunnel in R2L.
CATEGORY = {}
for c, subs in {
    "DoS": "back land neptune pod smurf teardrop apache2 mailbomb processtable udpstorm",
    "Probe": "ipsweep nmap portsweep satan mscan saint",
    "R2L": "ftp_write guess_passwd imap multihop phf warezmaster httptunnel named sendmail snmpgetattack snmpguess "
           "xlock xsnoop worm spy warezclient",
    "U2R": "buffer_overflow loadmodule perl rootkit ps sqlattack xterm",
}.items():
    for s in subs.split():
        CATEGORY[s] = c
CATEGORY["normal"] = "Normal"
CATS = ["DoS", "Probe", "R2L", "U2R"]
TEST_ONLY = set("apache2 mailbomb processtable udpstorm mscan saint httptunnel named sendmail snmpgetattack snmpguess "
                "xlock xsnoop worm ps sqlattack xterm".split())
TRAIN_ONLY = {"spy", "warezclient"}

# Features removed by FeCo, read off FeCo TDSC Fig. 4 (legend indices -> names):
# zero-var: num_outbound_cmds; Spearman: serror_rate(10), srv_serror_rate(17), rerror_rate(23) are absent from the
# ANOVA axis of Fig. 4(c); MI=0: land(32), root_shell(36), is_host_login(33); lowest ANOVA: num_compromised(20),
# num_root(11), dst_bytes(22).
FECO_PAPER_DROP = ["num_outbound_cmds", "serror_rate", "srv_serror_rate", "rerror_rate", "land", "root_shell",
                   "is_host_login", "num_compromised", "num_root", "dst_bytes"]


TEST_FULL = os.path.join(RAW, "KDDTest+_full.txt")  # official test file: 43 cols, sub-class labels
TEST_MIRROR = os.path.join(RAW, "KDDTest+.txt")  # binary-label Zenodo mirror; 306 rows corrupted (imap4 -> Attack4)
TEST_FILES = {"full": TEST_FULL, "mirror": TEST_MIRROR}


def file_md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_file_md5(test_file="full"):
    return file_md5(TEST_FILES[test_file])


def _read(path):
    """Handles 43-col (features,label,difficulty) and the 42-col binary-label test mirror (features, Attack/Normal)."""
    df = pd.read_csv(path, header=None, sep="\t" if "\t" in open(path).readline() else ",")
    if df.shape[1] == 43:
        df.columns = COLS
        df = df.drop(columns="difficulty")
    elif df.shape[1] == 42:
        df.columns = COLS[:42]
    else:
        raise ValueError(f"{path}: {df.shape[1]} columns")
    return df


def load_raw(test_file="full"):
    """Returns train/test frames with columns: 41 features, label, category, y, has_subclass.
    test_file='full' (default): KDDTest+_full.txt, 43 columns, 38 sub-classes.
    test_file='mirror': the binary-label mirror (306 attack rows have service='Attack4'); kept only to reproduce
    earlier CPU numbers (CPU-vs-GPU equivalence check). No silent fallback."""
    tr = _read(os.path.join(RAW, "KDDTrain+.txt"))
    path = TEST_FILES[test_file]
    if test_file == "full":
        ncol = pd.read_csv(path, header=None, nrows=5).shape[1]
        assert ncol == 43, f"{path}: {ncol} columns, expected 43"
    te = _read(path)
    assert len(tr) == 125973 and len(te) == 22544, (len(tr), len(te))
    if test_file == "full":
        assert not (te[CATEGORICAL].astype(str) == "Attack4").any().any(), "corrupted 'Attack4' value in test file"
    for df in (tr, te):
        lab = df.label.astype(str)
        if set(lab) <= {"Attack", "Normal"}:
            df["category"] = lab.map({"Attack": "Attack", "Normal": "Normal"})
            df["label"] = lab.map({"Attack": "attack", "Normal": "normal"})
            df.attrs["has_subclass"] = False
        else:
            unknown = set(lab) - set(CATEGORY)
            assert not unknown, f"unmapped labels {unknown}"
            df["category"] = lab.map(CATEGORY)
            df.attrs["has_subclass"] = True
        df["y"] = (df.label != "normal").astype(np.int64)
    return tr, te


def numeric_cols(df):
    return [c for c in COLS[:41] if c not in MI_FEATURES and c in df.columns]


# feature selection statistics (computable locally => federated)
def local_stats(df, y):
    """y: label used for MI/ANOVA (FeCo: 5-class category). Sufficient statistics a client can share: count, Spearman matrix (numeric), MI (binary/categorical),
    ANOVA F (numeric), variance. All computed on local data only."""
    num = numeric_cols(df)
    X = df[num].values.astype(np.float64)
    var = X.var(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = spearmanr(X).statistic if X.shape[0] > 2 else np.zeros((len(num), len(num)))
        F, _ = f_classif(X, y) if len(np.unique(y)) > 1 else (np.zeros(len(num)), None)
    rho = np.nan_to_num(np.asarray(rho), nan=0.0)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0)
    cat = df[MI_FEATURES].copy()
    for c in CATEGORICAL:
        cat[c] = cat[c].astype("category").cat.codes
    mi = (mutual_info_classif(cat.values, y, discrete_features=True, random_state=0)
          if len(np.unique(y)) > 1 else np.zeros(len(MI_FEATURES)))
    return {"n": len(df), "num": num, "var": var, "rho": rho, "F": F, "mi": mi}


def aggregate_stats(stats):
    """Server: count-weighted average of client statistics (federated feature selection)."""
    w = np.array([s["n"] for s in stats], dtype=np.float64)
    w = w / w.sum()
    agg = {"n": int(sum(s["n"] for s in stats)), "num": stats[0]["num"]}
    for k in ("var", "rho", "F", "mi"):
        agg[k] = sum(wi * s[k] for wi, s in zip(w, stats))
    return agg


def select_features(st, n_corr=3, n_anova=3, mi_eps=5e-3, log=None):
    """FeCo two-step selection from (possibly aggregated) statistics.
    1) drop zero-variance numeric; 2) greedily drop one feature from each of the top-n_corr |rho| pairs
    (the member with the lower ANOVA F); 3) drop MI<=mi_eps binary/categorical; 4) drop n_anova lowest-F numeric."""
    num = list(st["num"])
    drop, info = [], {}
    zv = [c for c, v in zip(num, st["var"]) if v <= 1e-12]
    drop += zv
    info["zero_var"] = zv
    idx = {c: i for i, c in enumerate(num)}
    alive = [c for c in num if c not in zv]
    R = np.abs(st["rho"])
    corr_drop = []
    for _ in range(n_corr):
        best, pair = -1, None
        for a_i, a in enumerate(alive):
            for b in alive[a_i + 1:]:
                r = R[idx[a], idx[b]]
                if r > best:
                    best, pair = r, (a, b)
        a, b = pair
        d = a if st["F"][idx[a]] < st["F"][idx[b]] else b
        corr_drop.append({"pair": [a, b], "rho": float(best), "dropped": d})
        alive.remove(d)
        drop.append(d)
    info["spearman"] = corr_drop
    mi_drop = [c for c, m in zip(MI_FEATURES, st["mi"]) if m <= mi_eps]
    info["mi"] = {c: float(m) for c, m in zip(MI_FEATURES, st["mi"])}
    info["mi_dropped"] = mi_drop
    drop += mi_drop
    Fs = sorted(alive, key=lambda c: st["F"][idx[c]])
    an_drop = Fs[:n_anova]
    info["anova_lowest"] = {c: float(st["F"][idx[c]]) for c in Fs[:n_anova + 3]}
    info["anova_dropped"] = an_drop
    drop += an_drop
    info["dropped"] = drop
    return drop, info


class Encoder:
    """One-hot (categories from train) + scaler fit on train only."""

    def __init__(self, drop, scaler="minmax", log1p=False):
        self.drop, self.scaler, self.log1p = list(drop), scaler, log1p

    def fit(self, df):
        keep = [c for c in COLS[:41] if c not in self.drop]
        self.cat = [c for c in CATEGORICAL if c in keep]
        self.num = [c for c in keep if c not in self.cat]
        self.levels = {c: sorted(df[c].unique()) for c in self.cat}
        X = self._num(df)
        if self.scaler == "minmax":
            self.a, self.b = X.min(0), X.max(0) - X.min(0)
        else:
            self.a, self.b = X.mean(0), X.std(0)
        self.b[self.b == 0] = 1.0
        return self

    def _num(self, df):
        X = df[self.num].values.astype(np.float64)
        return np.log1p(np.maximum(X, 0)) if self.log1p else X

    def transform(self, df):
        Xn = (self._num(df) - self.a) / self.b
        oh = [np.stack([(df[c].values == l) for l in self.levels[c]], 1).astype(np.float64) for c in self.cat]
        return np.concatenate([Xn] + oh, 1).astype(np.float32)

    @property
    def dim(self):
        return len(self.num) + sum(len(v) for v in self.levels.values())


def make_splits(tr, seed):
    """80/20 stratified (by sub-class label) train/val; val-benign split 50/50 into tune / calib."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(tr))
    strat = tr.label.where(tr.label.map(tr.label.value_counts()) >= 2, "rare").values
    i_tr, i_va = train_test_split(idx, test_size=0.2, stratify=strat, random_state=seed)
    vb = i_va[tr.y.values[i_va] == 0]
    rng = np.random.RandomState(seed)
    vb = rng.permutation(vb)
    calib = np.sort(vb[: len(vb) // 2])
    tune = np.sort(np.concatenate([vb[len(vb) // 2:], i_va[tr.y.values[i_va] == 1]]))
    return {"train": np.sort(i_tr), "val": np.sort(i_va), "val_tune": tune, "calib": calib}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tr, te = load_raw()
    for s in range(5):
        np.savez(os.path.join(OUT, f"splits_seed{s}.npz"), **make_splits(tr, s))
    st = local_stats(tr, tr.category.values)  # FeCo used the 5-class label for MI and ANOVA
    drop, info = select_features(st)
    info["feco_paper_drop"] = FECO_PAPER_DROP
    info["agree_with_paper"] = sorted(drop) == sorted(FECO_PAPER_DROP)
    enc = Encoder(drop).fit(tr)
    info["dim_procedure"] = enc.dim
    info["dim_paper"] = Encoder(FECO_PAPER_DROP).fit(tr).dim
    json.dump(info, open(os.path.join(OUT, "feature_selection_fulltrain.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))
