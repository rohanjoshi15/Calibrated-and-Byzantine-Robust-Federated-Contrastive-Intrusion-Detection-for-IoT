"""N-BaIoT: convert raw CSVs to per-device parquet (float32 features + labels) and audit counts.

Benign: data/raw/nbaiot/<dev>/benign_traffic.csv ; attacks: data/processed/nbaiot_extracted/<dev>/{gafgyt,mirai}_attacks/*.csv
Output: data/processed/nbaiot/<dev>.parquet with columns f0..f114 (float32), family (benign/gafgyt/mirai),
attack (benign/<name>), row_in_file (int, file order = time proxy; no timestamp column exists).
Run: python -m src.data.nbaiot
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from . import paths

RAW = os.path.join(paths.RAW, "nbaiot")
EXT = os.path.join(paths.PROCESSED, "nbaiot_extracted")
OUT = os.path.join(paths.PROCESSED, "nbaiot")
if os.path.isdir(RAW):
    DEVICES = sorted(d for d in os.listdir(RAW) if os.path.isdir(os.path.join(RAW, d)))
else:  # raw files absent: take the device list from the processed parquet files, if any
    DEVICES = sorted(f[:-8] for f in os.listdir(OUT) if f.endswith(".parquet")) if os.path.isdir(OUT) else []


def device_path(dev):
    return os.path.join(OUT, f"{dev}.parquet")


def convert_device(dev):
    parts, counts, cols0 = [], {}, None
    files = [("benign", "benign", os.path.join(RAW, dev, "benign_traffic.csv"))]
    for fam in ("gafgyt", "mirai"):
        for f in sorted(glob.glob(os.path.join(EXT, dev, f"{fam}_attacks", "*.csv"))):
            files.append((fam, os.path.splitext(os.path.basename(f))[0], f))
    for fam, att, f in files:
        df = pd.read_csv(f, dtype=np.float32)
        if cols0 is None:
            cols0 = list(df.columns)
        assert list(df.columns) == cols0, f"column mismatch in {f}"
        assert df.shape[1] == 115, f"{f} has {df.shape[1]} cols"
        n_nan = int(df.isna().sum().sum())
        n_inf = int(np.isinf(df.values).sum())
        df.columns = [f"f{i}" for i in range(115)]
        df["family"] = fam
        df["attack"] = att if fam == "benign" else f"{fam}_{att}"
        df["row_in_file"] = np.arange(len(df), dtype=np.int64)
        counts[f"{fam}/{att}"] = {"rows": len(df), "nan": n_nan, "inf": n_inf}
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full["family"] = full["family"].astype("category")
    full["attack"] = full["attack"].astype("category")
    full.to_parquet(device_path(dev), index=False)
    return counts, cols0


def load_device(dev, columns=None):
    return pd.read_parquet(device_path(dev), columns=columns)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    audit, names = {}, None
    for dev in DEVICES:
        c, cols = convert_device(dev)
        audit[dev] = c
        names = cols
        print(dev, sum(v["rows"] for v in c.values()), flush=True)
    with open(os.path.join(OUT, "feature_names.json"), "w") as f:
        json.dump(names, f)
    with open(os.path.join(OUT, "audit.json"), "w") as f:
        json.dump(audit, f, indent=1)
    tot = sum(v["rows"] for d in audit.values() for v in d.values())
    ben = sum(v["rows"] for d in audit.values() for k, v in d.items() if k.startswith("benign"))
    print("TOTAL", tot, "benign", ben, "malicious", tot - ben)
