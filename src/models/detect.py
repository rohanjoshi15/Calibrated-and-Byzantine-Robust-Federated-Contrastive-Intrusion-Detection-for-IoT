"""Benign templates/prototypes and scores on L2-normalised embeddings."""
import numpy as np
from scipy.special import ive, logsumexp


# Centroid accumulation: 'fast' = float64 sparse one-hot matmul (exact float64 sums, ~18x faster);
# 'legacy' = np.add.at into a float32 buffer (sequential float32 sums, as in the earlier CPU runs). Set per run by src.pipeline.run.
KMEANS_IMPL = "fast"


def scatter_sum(a, X, K, w=None, out_dtype=None):
    """Row sums of X (weighted by w) per label a ∈ [0, K): returns (K, d)."""
    if KMEANS_IMPL == "legacy":
        out = np.zeros((K, X.shape[1]), dtype=out_dtype or X.dtype)
        np.add.at(out, a, X if w is None else X * w[:, None])
        return out
    import scipy.sparse as sp
    n = len(a)
    M = sp.csr_matrix((np.ones(n) if w is None else np.asarray(w, float), (a, np.arange(n))), shape=(K, n))
    return np.asarray(M @ X.astype(np.float64))


def unit(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def template(Zb):
    """FeCo Eq. 5: mean of normalised benign embeddings (Zb already normalised)."""
    return Zb.mean(0)


def score_template(Z, zbar):
    """FeCo Eq. 6: cos(z̄, f(x))."""
    return unit(Z) @ unit(zbar)


def spherical_kmeans(Z, K, w=None, iters=50, seed=0, n_init=3):
    """Weighted spherical k-means (cosine). Z unit rows, w sample weights. Returns (C unit (K,o), counts (K,))."""
    rng = np.random.RandomState(seed)
    n = len(Z)
    w = np.ones(n) if w is None else np.asarray(w, float)
    K = min(K, n)
    best, best_obj = None, -np.inf
    for _ in range(n_init):
        # k-means++ style init on cosine distance
        C = [Z[rng.choice(n, p=w / w.sum())]]
        for _k in range(1, K):
            d = 1 - np.max(Z @ np.array(C).T, axis=1)
            p = np.maximum(d, 0) * w
            p = p / p.sum() if p.sum() > 0 else w / w.sum()
            C.append(Z[rng.choice(n, p=p)])
        C = np.array(C)
        for _it in range(iters):
            a = np.argmax(Z @ C.T, axis=1)
            Cn = scatter_sum(a, Z, len(C), w, out_dtype=C.dtype)
            empty = np.linalg.norm(Cn, axis=1) < 1e-12
            Cn[empty] = C[empty]
            Cn = unit(Cn).astype(C.dtype)
            if np.allclose(Cn, C):
                C = Cn
                break
            C = Cn
        a = np.argmax(Z @ C.T, axis=1)
        obj = float((w * (Z * C[a]).sum(1)).sum())
        if obj > best_obj:
            best_obj, best = obj, (C, np.bincount(a, weights=w, minlength=K))
    return best


def score_maxcos(Z, C):
    return np.max(unit(Z) @ C.T, axis=1)


# ---------------- vMF mixture ----------------
def log_cd(kappa, d):
    """log normaliser of vMF on S^{d-1}: C_d(κ) = κ^{d/2−1} / ((2π)^{d/2} I_{d/2−1}(κ)). Uses ive for stability."""
    nu = d / 2.0 - 1.0
    kappa = np.maximum(kappa, 1e-8)
    return nu * np.log(kappa) - (d / 2.0) * np.log(2 * np.pi) - (np.log(ive(nu, kappa)) + kappa)


def kappa_banerjee(rbar, d):
    rbar = np.clip(rbar, 1e-6, 1 - 1e-6)
    return rbar * (d - rbar ** 2) / (1 - rbar ** 2)


def fit_vmf_mixture(Z, K, iters=30, seed=0, w=None, kappa_max=1e4):
    """EM for a mixture of K vMFs (Banerjee et al., 2005), init from spherical k-means."""
    n, d = Z.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    C, cnt = spherical_kmeans(Z, K, w=w, seed=seed)
    K = len(C)
    pi = np.maximum(cnt, 1e-9) / cnt.sum()
    a = np.argmax(Z @ C.T, 1)
    kap = np.array([kappa_banerjee(np.linalg.norm((Z[a == k] * w[a == k, None]).sum(0)) / max(w[a == k].sum(), 1e-9), d)
                    for k in range(K)])
    kap = np.minimum(kap, kappa_max)
    for _ in range(iters):
        logp = np.log(pi)[None] + log_cd(kap, d)[None] + (Z @ C.T) * kap[None]
        R = np.exp(logp - logsumexp(logp, axis=1, keepdims=True)) * w[:, None]
        Nk = R.sum(0) + 1e-12
        pi = Nk / Nk.sum()
        S = R.T @ Z
        norm = np.linalg.norm(S, axis=1)
        C = S / np.maximum(norm[:, None], 1e-12)
        kap = np.minimum(kappa_banerjee(norm / Nk, d), kappa_max)
    return {"mu": C, "kappa": kap, "pi": pi}


def score_vmf(Z, m):
    d = Z.shape[1]
    logp = np.log(m["pi"])[None] + log_cd(m["kappa"], d)[None] + (unit(Z) @ m["mu"].T) * m["kappa"][None]
    return logsumexp(logp, axis=1)
