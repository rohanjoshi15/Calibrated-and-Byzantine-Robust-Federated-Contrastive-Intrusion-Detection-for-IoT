"""Load run results and select configs by validation metrics (never test)."""
import glob
import json
import os

import yaml


ROOT = os.environ.get("RESULTS_ROOT", "results")  # 'results/_v1' recomputes the earlier CPU numbers


def load_runs(phase, prefix="", max_seed=None):
    """All finished runs results/<phase>/<prefix>*. max_seed: keep seeds < max_seed (seeds 5-9 exist only for
    the H1 study, so the other tables use seeds 0-4)."""
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, phase, prefix + "*"))):
        mp = os.path.join(d, "metrics.json")
        if os.path.exists(mp):
            c = yaml.safe_load(open(os.path.join(d, "config.yaml")))
            if max_seed is not None and c.get("seed", 0) >= max_seed:
                continue
            out.append((c, json.load(open(mp))))
    return out


def device_of(runs):
    """Device label of a table cell: 'cuda' / 'cpu' (v1 runs have no device key) / 'mixed:...'."""
    ds = sorted({m.get("device", "cpu") for _, m in runs})
    return ds[0] if len(ds) == 1 else "mixed:" + "+".join(ds)


def get(d, path, default=None):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def val_key(m, det="template"):
    """Selection objective: val TPR@1%FPR, tie-break val AUROC."""
    r = get(m, f"results.{det}.val.ranking", {}) or {}
    return (round(r.get("tpr@1", -1), 4), r.get("auroc", -1))


def best(runs, det="template"):
    return max(runs, key=lambda cm: val_key(cm[1], det))
