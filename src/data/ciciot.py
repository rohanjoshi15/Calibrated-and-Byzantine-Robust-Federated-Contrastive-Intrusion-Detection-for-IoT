"""CICIoT2023 (per-pcap CSV subset): convert to one parquet with labels from folder names.

Each class folder holds *.pcap.csv files with 39 features and no label column (README_CSV.pdf: label is a
feature only in the merged CSVs). Output: data/processed/ciciot2023/all.parquet + audit.json.
Run: python -m src.data.ciciot
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from . import paths

RAW = os.path.join(paths.RAW, "ciciot2023")
OUT = os.path.join(paths.PROCESSED, "ciciot2023")

FAMILY = {
    "Benign_Final": "Benign",
    "DDoS-ICMP_Flood": "DDoS", "DDoS-SYN_Flood": "DDoS", "DDoS-UDP_Flood": "DDoS",
    "DoS-SYN_Flood": "DoS", "DoS-UDP_Flood": "DoS",
    "Mirai-greeth_flood": "Mirai", "Mirai-udpplain": "Mirai",
    "Recon-OSScan": "Recon", "Recon-PortScan": "Recon",
    "MITM-ArpSpoofing": "Spoofing", "DNS_Spoofing": "Spoofing",
    "SqlInjection": "Web", "XSS": "Web", "CommandInjection": "Web",
    "DictionaryBruteForce": "BruteForce",
}


def load(columns=None):
    return pd.read_parquet(os.path.join(OUT, "all.parquet"), columns=columns)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    parts, audit, cols0 = [], {}, None
    for cls in sorted(os.listdir(RAW)):
        d = os.path.join(RAW, cls)
        if not os.path.isdir(d):
            continue
        assert cls in FAMILY, f"unmapped class {cls}"
        for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
            df = pd.read_csv(f)
            if cols0 is None:
                cols0 = list(df.columns)
            assert list(df.columns) == cols0, f"col mismatch {f}"
            df = df.astype(np.float32)
            x = df.values
            audit.setdefault(cls, {"family": FAMILY[cls], "files": {}, "rows": 0})
            audit[cls]["files"][os.path.basename(f)] = {
                "rows": len(df), "nan": int(np.isnan(x).sum()), "inf": int(np.isinf(x).sum())}
            audit[cls]["rows"] += len(df)
            df["cls"] = cls
            df["family"] = FAMILY[cls]
            df["file"] = os.path.basename(f)
            df["row_in_file"] = np.arange(len(df), dtype=np.int64)
            parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    for c in ("cls", "family", "file"):
        full[c] = full[c].astype("category")
    full.columns = [c.strip() for c in full.columns]
    full.to_parquet(os.path.join(OUT, "all.parquet"), index=False)
    fam = full.groupby("family", observed=True).size().to_dict()
    json.dump({"features": [c.strip() for c in cols0], "classes": audit, "family_rows": fam,
               "small_classes_lt5k": [c for c, v in audit.items() if v["rows"] < 5000]},
              open(os.path.join(OUT, "audit.json"), "w"), indent=1)
    print(fam)
    print("classes <5k:", [c for c, v in audit.items() if v["rows"] < 5000])
