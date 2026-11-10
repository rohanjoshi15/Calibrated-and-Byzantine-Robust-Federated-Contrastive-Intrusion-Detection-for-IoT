"""FL partitioners. Inputs: y (0 benign / 1 attack), cat (attack category per sample; ignored for benign).
Return: list of index arrays (one per client) into the given arrays.

- iid: random equal shards of every class (FeCo: "randomly select a subset of each class").
- noniid1: client i holds n_i ~ U{1..n_max} attack categories (FeCo non-IID-1, n ∈ {1,2,3,4}).
- noniid2: exactly 1 category per client, assigned round-robin over a shuffled list (each category covered).
- dirichlet: for every attack category c, client proportions ~ Dir(α·1).
In all cases benign data is split into equal random shards (FeCo: all clients hold benign), unless benign_groups is
given with benign_alpha: then benign is split by Dirichlet over groups (benign heterogeneity, used for H3).
no_attack_frac: that share of clients gets its attack samples removed (redistributed to others)."""
import numpy as np


def _split_equal(idx, n, rng):
    idx = rng.permutation(idx)
    return np.array_split(idx, n)


def _benign(yb_idx, n, rng, groups=None, alpha=None):
    if groups is None or alpha is None:
        return _split_equal(yb_idx, n, rng)
    parts = [[] for _ in range(n)]
    for g in np.unique(groups[yb_idx]):
        ig = rng.permutation(yb_idx[groups[yb_idx] == g])
        p = rng.dirichlet(alpha * np.ones(n))
        cuts = (np.cumsum(p) * len(ig)).astype(int)[:-1]
        for i, s in enumerate(np.split(ig, cuts)):
            parts[i].append(s)
    out = [np.concatenate(p) if p else np.array([], int) for p in parts]
    # guarantee a minimum benign share per client (needed for local training/calibration): top up from the largest
    min_n = max(20, len(yb_idx) // (n * 20))
    for i in range(n):
        while len(out[i]) < min_n:
            j = int(np.argmax([len(o) for o in out]))
            take = min(min_n - len(out[i]), len(out[j]) - min_n)
            if take <= 0:
                break
            out[i] = np.concatenate([out[i], out[j][:take]])
            out[j] = out[j][take:]
    return out


def partition(y, cat, n_clients, scheme, seed, n_max=4, alpha=0.5, no_attack_frac=0.0, benign_groups=None,
              benign_alpha=None):
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    cat = np.asarray(cat)
    ib = np.flatnonzero(y == 0)
    ia = np.flatnonzero(y == 1)
    cats = sorted(np.unique(cat[ia]).tolist())
    ben = _benign(ib, n_clients, rng, benign_groups, benign_alpha)
    att = [[] for _ in range(n_clients)]
    attackers = np.arange(n_clients)
    if no_attack_frac > 0:
        k = int(round(no_attack_frac * n_clients))
        attackers = np.sort(rng.permutation(n_clients)[k:])
    m = len(attackers)
    if scheme == "iid":
        for c in cats:
            for i, s in zip(attackers, _split_equal(ia[cat[ia] == c], m, rng)):
                att[i].append(s)
    elif scheme in ("noniid1", "noniid2"):
        if scheme == "noniid2":
            order = np.concatenate([rng.permutation(cats) for _ in range(int(np.ceil(m / len(cats))))])[:m]
            hold = {i: [order[j]] for j, i in enumerate(attackers)}
        else:
            hold = {}
            for i in attackers:
                n = rng.randint(1, min(n_max, len(cats)) + 1)
                hold[i] = list(rng.choice(cats, n, replace=False))
            # ensure every category is held by someone
            for c in cats:
                if not any(c in h for h in hold.values()):
                    hold[attackers[rng.randint(m)]].append(c)
        for c in cats:
            owners = [i for i in attackers if c in hold[i]]
            for i, s in zip(owners, _split_equal(ia[cat[ia] == c], len(owners), rng)):
                att[i].append(s)
    elif scheme == "dirichlet":
        for c in cats:
            ic = rng.permutation(ia[cat[ia] == c])
            p = rng.dirichlet(alpha * np.ones(m))
            cuts = (np.cumsum(p) * len(ic)).astype(int)[:-1]
            for i, s in zip(attackers, np.split(ic, cuts)):
                att[i].append(s)
    else:
        raise ValueError(scheme)
    out = []
    for i in range(n_clients):
        a = np.concatenate(att[i]) if att[i] else np.array([], int)
        out.append(np.sort(np.concatenate([ben[i], a]).astype(int)))
    return out


def describe(parts, y, cat):
    rows = []
    for p in parts:
        yy, cc = y[p], cat[p]
        rows.append({"n": len(p), "benign": int((yy == 0).sum()),
                     **{str(c): int(((cc == c) & (yy == 1)).sum()) for c in np.unique(cat[y == 1])}})
    return rows
