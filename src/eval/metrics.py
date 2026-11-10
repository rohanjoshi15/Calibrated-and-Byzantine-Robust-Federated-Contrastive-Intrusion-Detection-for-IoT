"""Detection metrics. Convention: score s = similarity (higher = benign); anomaly score = −s; positive class = attack."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def binary(y, pred):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(len(y), 1)
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    fpr = fp / max(fp + tn, 1)
    return {"acc": acc, "recall": rec, "precision": prec, "f1": f1, "fpr": fpr, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def roc(y, anom, drop_intermediate=False):
    """All ROC points (sklearn's default drop_intermediate=True removes collinear points, which with tied
    scores can hide the best operating point below an FPR target)."""
    return roc_curve(y, anom, drop_intermediate=drop_intermediate)


def tpr_at_fpr(y, anom, target, drop_intermediate=False):
    """Max TPR with FPR ≤ target (step ROC, no interpolation)."""
    fpr, tpr, _ = roc(y, anom, drop_intermediate)
    ok = fpr <= target + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def fpr_at_tpr(y, anom, target, drop_intermediate=False):
    """Min FPR achieving TPR ≥ target (FeCo Fig. 9 reports FPR at recall fixed to 0.9998)."""
    fpr, tpr, _ = roc(y, anom, drop_intermediate)
    ok = tpr >= target - 1e-12
    return float(fpr[ok].min()) if ok.any() else 1.0


def ranking(y, s, legacy=True):
    """Threshold-free metrics. AUROC/AUPRC use sklearn's exact formulas (no point dropping). legacy=True also stores
    the drop_intermediate=True TPR values under '*_v1dropint' so the size of the difference can be measured."""
    y = np.asarray(y).astype(int)
    anom = -np.asarray(s, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return {}
    out = {"auroc": float(roc_auc_score(y, anom)), "auprc": float(average_precision_score(y, anom))}
    fpr, tpr, _ = roc(y, anom, False)
    for t, k in ((0.01, "tpr@1"), (0.05, "tpr@5")):
        ok = fpr <= t + 1e-12
        out[k] = float(tpr[ok].max()) if ok.any() else 0.0
    ok = tpr >= 0.9998 - 1e-12
    out["fpr@tpr99.98"] = float(fpr[ok].min()) if ok.any() else 1.0
    if legacy:
        out["tpr@1_v1dropint"] = tpr_at_fpr(y, anom, 0.01, True)
        out["tpr@5_v1dropint"] = tpr_at_fpr(y, anom, 0.05, True)
    return out


def per_group_recall(pred, groups, y):
    """FeCo Eq. 10: fraction predicted intrusion per attack group (ignores benign)."""
    out = {}
    for g in np.unique(groups[y == 1]):
        m = (groups == g) & (y == 1)
        out[str(g)] = float(np.mean(pred[m])) if m.any() else float("nan")
    return out


def evaluate(y, s, rho, groups=None, novel=None, rank=None):
    """rank: precomputed ranking(y, s) (identical for every threshold on the same scores) to avoid recomputation."""
    pred = (np.asarray(s) < rho).astype(int)
    m = {**binary(y, pred), **(ranking(y, s) if rank is None else rank), "rho": float(rho)}
    if groups is not None:
        m["recall_by_group"] = per_group_recall(pred, np.asarray(groups), np.asarray(y))
    if novel is not None and np.asarray(novel).any():
        m["novel_recall"] = float(pred[np.asarray(novel)].mean())
    return m


def fpr_violation(fprs, ns, alpha):
    """Per-client |FPR−α| and share with FPR > α + 1/(n_i+1)."""
    fprs, ns = np.asarray(fprs, float), np.asarray(ns, float)
    return {"mean_abs_fpr_minus_alpha": float(np.mean(np.abs(fprs - alpha))),
            "frac_violate": float(np.mean(fprs > alpha + 1.0 / (ns + 1))),
            "frac_above_alpha": float(np.mean(fprs > alpha)),
            "mean_fpr": float(np.mean(fprs)), "max_fpr": float(np.max(fprs)), "std_fpr": float(np.std(fprs))}


def mean_std(dicts, keys=None):
    keys = keys or [k for k, v in dicts[0].items() if isinstance(v, (int, float))]
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in dicts if d.get(k) is not None], float)
        if len(vals):
            out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, "n": len(vals)}
    return out
