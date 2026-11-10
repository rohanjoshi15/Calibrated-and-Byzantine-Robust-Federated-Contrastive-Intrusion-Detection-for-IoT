"""Thresholds on a similarity score s (higher = more benign; intrusion iff s < ρ).

(a) feco_percentile: FeCo Eq. 7 on benign TRAINING scores: r = ⌊p/100·N⌋, ρ = S_r (1-indexed, ascending).
(b) conformal: split-conformal on held-out benign calibration scores.
    'floor' (default, valid): k = ⌊α(n+1)⌋, ρ = s_(k)  ⇔ flag iff conformal p-value p(x) = (1+#{s_i ≤ s(x)})/(n+1) ≤ α
            ⇒ P(FPR event) = k/(n+1) ≤ α (exchangeability). k=0 ⇒ ρ = −inf (never flag).
    'ceil': k = ⌈α(n+1)⌉ ⇒ marginal FPR = k/(n+1) ∈ [α, α+1/(n+1)).
(c) federated global quantile from summed client histograms (fixed bins on [−1, 1]); conservative: returns the lower
    edge of the bin containing the k-th order statistic, so global FPR ≤ that of the exact pooled quantile.
(d) robust quantile (Rob-FCP style) lives in src/calib/robust.py.
"""
import math

import numpy as np


def feco_percentile(train_benign_scores, p=5.0):
    S = np.sort(np.asarray(train_benign_scores))
    r = int(math.floor(p / 100.0 * len(S)))
    return float(S[max(r - 1, 0)])


def conformal_k(n, alpha, mode="floor"):
    k = math.floor(alpha * (n + 1) + 1e-9) if mode == "floor" else math.ceil(alpha * (n + 1) - 1e-9)
    return min(k, n)


def conformal(cal_scores, alpha, mode="floor"):
    S = np.sort(np.asarray(cal_scores))
    k = conformal_k(len(S), alpha, mode)
    if k < 1:
        return -np.inf
    return float(S[k - 1])


BINS = np.linspace(-1.0, 1.0, 4001)  # bin width 5e-4


def histogram(scores, bins=BINS):
    return np.histogram(np.clip(scores, bins[0], bins[-1]), bins=bins)[0].astype(np.int64)


def from_histogram(hist, alpha, bins=BINS, mode="floor"):
    n = int(hist.sum())
    k = conformal_k(n, alpha, mode)
    if k < 1:
        return -np.inf
    cum = np.cumsum(hist)
    b = int(np.searchsorted(cum, k))  # first bin with cum >= k
    return float(bins[b])  # lower edge: s < lower edge ⇒ s strictly below the k-th order stat bin


def global_quantile(client_scores, alpha, mode="floor"):
    hist = sum(histogram(s) for s in client_scores)
    return from_histogram(hist, alpha, mode=mode)
