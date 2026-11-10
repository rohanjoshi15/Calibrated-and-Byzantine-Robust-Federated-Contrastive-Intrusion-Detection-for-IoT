"""Aggregation rules on flattened parameter vectors (rows = clients)."""
import numpy as np
import torch


def flatten(model):
    return torch.cat([p.detach().reshape(-1) for p in model.state_dict().values()]).clone()


def load_flat(model, vec):
    sd, i = model.state_dict(), 0
    new = {}
    for k, v in sd.items():
        n = v.numel()
        new[k] = vec[i:i + n].reshape(v.shape).to(v.dtype)
        i += n
    model.load_state_dict(new)
    return model


def fedavg(W, n):
    """FeCo Eq. 9 / FedAvg: weights c_i = n_i / Σ n."""
    w = torch.as_tensor(np.asarray(n, float) / np.sum(n), dtype=W.dtype, device=W.device)
    return (W * w[:, None]).sum(0)


def trimmed_mean(W, n=None, beta=0.2):
    """Coordinate-wise trimmed mean, trims ⌊β m⌋ from each side (Yin et al. 2018)."""
    m = W.shape[0]
    k = int(np.floor(beta * m))
    if k == 0 or m - 2 * k < 1:
        return W.mean(0)
    S, _ = torch.sort(W, dim=0)
    return S[k:m - k].mean(0)


def median(W, n=None):
    return W.median(dim=0).values if W.shape[0] % 2 else torch.quantile(W, 0.5, dim=0)


def geometric_median(W, n=None, iters=100, eps=1e-6):
    """Weiszfeld (smoothed) geometric median of rows."""
    z = W.mean(0)
    w = (torch.ones(W.shape[0], dtype=W.dtype, device=W.device) if n is None
         else torch.as_tensor(np.asarray(n, float), dtype=W.dtype, device=W.device))
    for _ in range(iters):
        d = torch.norm(W - z, dim=1).clamp_min(eps)
        a = w / d
        z_new = (W * a[:, None]).sum(0) / a.sum()
        if torch.norm(z_new - z) < eps * (1 + torch.norm(z)):
            z = z_new
            break
        z = z_new
    return z


AGG = {"fedavg": fedavg, "trimmed": trimmed_mean, "median": median, "geomed": geometric_median}
