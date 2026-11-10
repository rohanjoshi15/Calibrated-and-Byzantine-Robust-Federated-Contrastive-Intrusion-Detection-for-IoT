"""Migration for runs with data.train_on='full' (FeCo protocol): the NSL-KDD calib rows are inside the training set,
so the stored split-conformal metrics are invalid. Renames every key 'conf_a*' → 'invalid_conf_a*' inside
metrics.json (results.*.rho / val / test / zeroday and results_per_client). An unmodified copy of every
metrics.json must already exist under results/_v1/ (the script refuses otherwise). Idempotent.
python experiments/migrate_b5.py [--root results]"""
import argparse
import glob
import json
import os
import re

import yaml

PAT = re.compile(r"^conf_a")


def rename(o):
    if isinstance(o, dict):
        return {("invalid_" + k if isinstance(k, str) and PAT.match(k) else k): rename(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rename(v) for v in o]
    return o


def main(root):
    n = 0
    for cfgp in sorted(glob.glob(os.path.join(root, "phase*", "*", "config.yaml"))):
        cfg = yaml.safe_load(open(cfgp))
        if (cfg.get("data") or {}).get("train_on") != "full":
            continue
        d = os.path.dirname(cfgp)
        mp = os.path.join(d, "metrics.json")
        if not os.path.exists(mp):
            continue
        rel = os.path.relpath(mp, root)
        v1 = os.path.join(root, "_v1", rel)
        assert os.path.exists(v1), f"no v1 copy for {mp}; archive first"
        m = json.load(open(mp))
        m2 = rename(m)
        if m2 != m:
            m2["migration_b5"] = "conf_a* renamed to invalid_conf_a* (calib ⊂ train when train_on='full'); original in results/_v1"
            tmp = mp + ".tmp"
            json.dump(m2, open(tmp, "w"), indent=1)
            os.replace(tmp, mp)
            n += 1
    print(f"migrated {n} metrics.json files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    main(ap.parse_args().root)
