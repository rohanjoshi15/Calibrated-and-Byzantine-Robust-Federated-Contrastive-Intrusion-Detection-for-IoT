"""FeCo contrastive loss (TDSC Eq. 2-4) and optional prototype-compactness term (C1)."""
import torch
import torch.nn.functional as F


_OFFDIAG = {}


def _offdiag(nb, device, dtype):
    k = (nb, str(device), dtype)
    if k not in _OFFDIAG:
        if len(_OFFDIAG) > 64:
            _OFFDIAG.clear()
        _OFFDIAG[k] = 1.0 - torch.eye(nb, device=device, dtype=dtype)
    return _OFFDIAG[k]


def split_batch(z, y, nb=None):
    """(benign rows, attack rows). nb given ⇔ the batch is ordered benign-first (stratified sampler): slicing selects
    exactly the rows z[y==0], z[y==1] in the same order, without the device→host sync of boolean indexing."""
    if nb is not None:
        return z[:nb], z[nb:]
    return z[y == 0], z[y == 1]


def feco_loss(z, y, tau=0.1, normalize=True, nb=None):
    """z: (B, o) embeddings, y: (B,) 0=benign 1=attack.
    l_ij = exp(s_ij/τ) / [exp(s_ij/τ) + Σ_m exp(s_im/τ)], s = v_iᵀv_j (cosine if normalize), u_m attacks.
    L = mean over ordered benign pairs i≠j of −log l_ij  (== Eq. 4: 1/(N(N−1)) Σ_{i<j} (L_ij + L_ji)).
    Computed with logsumexp. Returns 0 (with grad) if <2 benign. With no attacks, l_ij = 1 → loss 0.
    nb: benign-first batch (see split_batch); the off-diagonal mean is then Σ(nll·(1−I))/(nb(nb−1)): same value,
    float32 summation order may differ in the last bits from the masked mean."""
    if normalize:
        z = F.normalize(z, dim=1)
    v, u = split_batch(z, y, nb)
    nbv = v.shape[0]
    if nbv < 2:
        return z.sum() * 0.0
    s_vv = v @ v.T / tau  # (nb, nb)
    if u.shape[0] > 0:
        neg = torch.logsumexp(v @ u.T / tau, dim=1, keepdim=True)  # (nb, 1)
        denom = torch.logaddexp(s_vv, neg.expand_as(s_vv))
    else:
        denom = s_vv
    nll = denom - s_vv  # −log l_ij
    if nb is not None:
        return (nll * _offdiag(nbv, z.device, nll.dtype)).sum() / (nbv * (nbv - 1))
    mask = ~torch.eye(nbv, dtype=torch.bool, device=z.device)
    return nll[mask].mean()


def proto_loss(z, y, protos, nb=None):
    """λ-term: mean over benign of (1 − max_k cos(v_i, c_k)). protos: (K, o) unit vectors (no grad)."""
    v = F.normalize(split_batch(z, y, nb)[0], dim=1)
    if v.shape[0] == 0 or protos is None:
        return z.sum() * 0.0
    return (1.0 - (v @ protos.T).max(dim=1).values).mean()


def svdd_loss(z, y, c, eta=1.0, eps=1e-6, nb=None):
    """Deep SVDD (y ignored, benign only) / Deep SAD (Ruff et al. 2020, Eq. 3): benign ‖z−c‖², labelled
    anomalies η(‖z−c‖²)^{-1}."""
    d2 = ((z - c) ** 2).sum(1)
    ben, att = split_batch(d2, y, nb)
    loss = ben.sum()
    if att.numel() > 0 and eta > 0:
        loss = loss + eta * (1.0 / (att + eps)).sum()
    return loss / max(len(d2), 1)


def clad_loss(z, y, nb=None):
    """CLAD (Wilkie et al., TNSM 2026, Eq. 7): over benign anchors i,
    mean_p d(z_i,z_p)^2 + mean_n (1 − d(z_i,z_n))^2, d(z,z') = (1 − zᵀz')/2, z L2-normalised.
    P(i) = other benign samples in the batch, N(i) = attack samples."""
    z = F.normalize(z, dim=1)
    v, u = split_batch(z, y, nb)
    nb = v.shape[0]
    if nb < 2:
        return z.sum() * 0.0
    dvv = (1 - v @ v.T) / 2
    mask = _offdiag(nb, z.device, dvv.dtype)
    pos = (dvv ** 2 * mask).sum(1) / (nb - 1)
    if u.shape[0] > 0:
        neg = ((1 - (1 - v @ u.T) / 2) ** 2).mean(1)
    else:
        neg = torch.zeros_like(pos)
    return (pos + neg).mean()
