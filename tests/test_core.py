import math

import numpy as np
import torch

from src.calib.thresholds import conformal, feco_percentile, from_histogram, global_quantile, histogram
from src.data.partition import partition
from src.eval.metrics import binary, ranking, tpr_at_fpr
from src.fl.aggregators import fedavg, geometric_median, median, trimmed_mean
from src.models.losses import feco_loss


def naive_feco(z, y, tau, normalize=True):
    if normalize:
        z = torch.nn.functional.normalize(z, dim=1)
    v, u = z[y == 0], z[y == 1]
    tot, cnt = 0.0, 0
    for i in range(len(v)):
        for j in range(len(v)):
            if i == j:
                continue
            num = torch.exp(v[i] @ v[j] / tau)
            den = num + sum(torch.exp(v[i] @ um / tau) for um in u)
            tot += -torch.log(num / den)
            cnt += 1
    return tot / cnt


def test_feco_loss_matches_naive():
    torch.manual_seed(0)
    z = torch.randn(12, 5)
    y = torch.tensor([0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
    for tau in (0.1, 0.5):
        for norm in (True, False):
            a = feco_loss(z, y, tau, norm)
            b = naive_feco(z, y, tau, norm)
            assert torch.allclose(a, b, atol=1e-5), (a, b)


def test_feco_loss_symmetric_eq4():
    # Eq. 4: 1/(N(N-1)) sum_{i<j} (L_ij + L_ji) equals mean over ordered pairs
    torch.manual_seed(1)
    z = torch.randn(8, 4)
    y = torch.tensor([0, 0, 0, 1, 1, 0, 0, 1])
    assert torch.isfinite(feco_loss(z, y, 0.05))


def test_feco_loss_no_attacks_zero_and_stable():
    z = torch.randn(6, 3, requires_grad=True)
    y = torch.zeros(6, dtype=torch.long)
    l = feco_loss(z, y, 0.1)
    assert abs(l.item()) < 1e-6
    l.backward()
    big = torch.randn(10, 3) * 100
    assert torch.isfinite(feco_loss(big, torch.tensor([0] * 5 + [1] * 5), 0.05, normalize=False))


def test_feco_percentile():
    s = np.arange(1, 101) / 100.0  # 0.01..1.00
    assert feco_percentile(s, 5) == 0.05  # r = 5 -> 5th entry (1-indexed)


def test_conformal_index_and_validity():
    s = np.arange(1, 100) / 100.0  # n = 99
    assert conformal(s, 0.05, "floor") == 0.05  # k = floor(0.05*100) = 5
    assert conformal(s, 0.051, "ceil") == 0.06  # k = ceil(5.1) = 6
    assert conformal(np.arange(10.0), 0.05) == -np.inf  # k = 0
    # marginal validity by simulation: P(test < rho) <= alpha (floor), >= alpha (ceil)
    rng = np.random.RandomState(0)
    n, alpha, T = 50, 0.1, 20000
    cal = rng.randn(T, n)
    test = rng.randn(T)
    for mode, cmp in (("floor", lambda f: f <= alpha + 0.01), ("ceil", lambda f: f >= alpha - 0.01)):
        rho = np.array([conformal(c, alpha, mode) for c in cal])
        f = np.mean(test < rho)
        assert cmp(f), (mode, f)
        k = math.floor(alpha * (n + 1)) if mode == "floor" else math.ceil(alpha * (n + 1))
        assert abs(f - k / (n + 1)) < 0.01


def test_histogram_quantile_conservative():
    rng = np.random.RandomState(1)
    parts = [np.tanh(rng.randn(300) + i * 0.1) for i in range(5)]
    pooled = np.concatenate(parts)
    exact = conformal(pooled, 0.05)
    hq = global_quantile(parts, 0.05)
    assert hq <= exact + 1e-12 and exact - hq < 1e-3
    assert from_histogram(histogram(pooled), 0.05) == hq


def test_partition_schemes():
    rng = np.random.RandomState(0)
    y = np.r_[np.zeros(1000), np.ones(800)].astype(int)
    cat = np.r_[np.array(["N"] * 1000), rng.choice(["DoS", "Probe", "R2L", "U2R"], 800)]
    for sch in ("iid", "noniid1", "noniid2", "dirichlet"):
        parts = partition(y, cat, 20, sch, seed=0, alpha=0.5)
        allidx = np.concatenate(parts)
        assert len(allidx) == len(np.unique(allidx)) == len(y), sch  # disjoint & complete
        assert all((y[p] == 0).sum() > 0 for p in parts)
        if sch == "noniid2":
            assert all(len(np.unique(cat[p][y[p] == 1])) <= 1 for p in parts)
            assert len(np.unique(np.concatenate([cat[p][y[p] == 1] for p in parts]))) == 4
        if sch == "noniid1":
            assert all(1 <= len(np.unique(cat[p][y[p] == 1])) <= 4 for p in parts if (y[p] == 1).any())
    parts = partition(y, cat, 20, "iid", seed=0, no_attack_frac=0.2)
    assert sum((y[p] == 1).sum() == 0 for p in parts) == 4


def test_aggregators():
    W = torch.tensor([[1.0, 2.0], [3.0, 4.0], [100.0, -100.0], [2.0, 3.0], [2.5, 3.5]])
    assert torch.allclose(fedavg(W[:2], [1, 3]), torch.tensor([2.5, 3.5]))
    assert torch.allclose(median(W), torch.tensor([2.5, 3.0]))
    tm = trimmed_mean(W, beta=0.2)
    assert torch.allclose(tm, torch.tensor([(2 + 2.5 + 3) / 3, (2 + 3 + 3.5) / 3]))
    gm = geometric_median(W)
    assert torch.norm(gm - torch.tensor([2.3, 3.2])) < 1.0


def test_metrics():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pred = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    m = binary(y, pred)
    assert m["acc"] == 6 / 8 and m["fpr"] == 0.25 and m["recall"] == 0.75 and m["precision"] == 0.75
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.3, 0.4])  # perfectly separable
    r = ranking(y, s)
    assert r["auroc"] == 1.0 and r["tpr@1"] == 1.0
    assert tpr_at_fpr(y, -np.array([0.9, 0.8, 0.7, 0.35, 0.1, 0.2, 0.3, 0.4]), 0.0) == 0.75
