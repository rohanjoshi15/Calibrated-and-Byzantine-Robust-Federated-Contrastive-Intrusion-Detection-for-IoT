"""Robust federated quantile following Rob-FCP (Kang et al., ICML 2024, §3.2, Eq. 5-6).

Each client k reports a characterization vector v(k) = normalised histogram of its calibration scores (H bins) and its
count n_k. Server: pairwise ℓp distances d(k1,k2) = ‖v(k1)−v(k2)‖_p; maliciousness M(k) = mean distance to the
K_b−1 nearest neighbours; benign set = K_b clients with lowest M; global threshold from the count-weighted pooled
histogram of the benign set. K_b = K − ⌈f_max·K⌉ with f_max the assumed upper bound on the malicious fraction
(Rob-FCP also gives a likelihood-based estimate of K_b; not implemented)."""
import math

import numpy as np

from .thresholds import BINS, from_histogram


def char_vectors(hists, H=100):
    """Coarsen fine histograms (len(BINS)-1 bins) into H bins and normalise."""
    out = []
    for h in hists:
        h = np.asarray(h, float)
        g = h.reshape(H, -1).sum(1) if len(h) % H == 0 else np.histogram(np.repeat(np.arange(len(h)), h.astype(int)), bins=H)[0]
        out.append(g / max(g.sum(), 1))
    return np.array(out)


def maliciousness(V, Kb, p=2):
    D = np.linalg.norm(V[:, None, :] - V[None, :, :], ord=p, axis=2)
    np.fill_diagonal(D, np.inf)
    t = max(Kb - 1, 1)
    return np.sort(D, axis=1)[:, :t].mean(1)


def robust_quantile(hists, alpha, f_max=0.3, H=100, p=2, mode="floor"):
    """hists: list of fine histograms over BINS (counts). Returns (rho, benign_ids, M)."""
    K = len(hists)
    Kb = K - int(math.ceil(f_max * K))
    V = char_vectors(hists, H=H)
    M = maliciousness(V, Kb, p=p)
    keep = np.argsort(M)[:Kb]
    pooled = np.sum([hists[i] for i in keep], axis=0)
    return from_histogram(pooled, alpha, BINS, mode), np.sort(keep), M
