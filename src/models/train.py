"""Local/centralised training of the FeCo encoder with a stratified benign/attack batch sampler.

One epoch = one pass over the benign anchors. Each batch has Bb = round(B·(1−a)) benign + Ba = B−Bb attack samples
(a = attack_frac); attack indices are cycled through random permutations (with repetition if few attacks).
Supports FedProx (μ/2‖w−w_g‖²), MOON (model-contrastive term), prototype compactness (λ), and a Byzantine
"flip" mode (attacker trains with its botnet flows labelled benign: whitelisting poisoning)."""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from .losses import clad_loss, feco_loss, proto_loss, svdd_loss


def default_cfg():
    return dict(epochs=4, batch_size=512, attack_frac=0.5, lr=1e-3, momentum=0.9, weight_decay=0.0, tau=0.1,
                normalize=True, mu_prox=0.0, mu_moon=0.0, tau_moon=0.5, lam_proto=0.0, grad_clip=10.0)


def batches(yb, rng, B, a):
    if a == "random":  # unstratified: shuffle all samples, natural class mix per batch; epoch = pass over all data
        perm = rng.permutation(len(yb))
        for s in range(0, len(perm), B):
            bi = perm[s:s + B]
            if (yb[bi] == 0).sum() >= 2:
                yield bi, int((yb[bi] == 0).sum())
        return
    ib = np.flatnonzero(yb == 0)
    ia = np.flatnonzero(yb == 1)
    if len(ib) < 2:
        return
    Ba = int(round(B * a)) if len(ia) > 0 else 0
    Bb = B - Ba
    perm_b = rng.permutation(ib)
    pa, pos = rng.permutation(ia) if len(ia) else ia, 0
    for s in range(0, len(perm_b), Bb):
        bb = perm_b[s:s + Bb]
        if len(bb) < 2:
            continue
        if Ba:
            take = []
            need = Ba
            while need > 0:
                if pos >= len(pa):
                    pa, pos = rng.permutation(ia), 0
                t = pa[pos:pos + need]
                take.append(t)
                pos += len(t)
                need -= len(t)
            yield np.concatenate([bb] + take), len(bb)
        else:
            yield bb, len(bb)


def train_local(model, X, y, cfg, rng, global_model=None, prev_model=None, protos=None,
                train_y=None, epoch_cb=None):
    """X: float32 tensor (n, d); y: np int array (0/1). train_y: labels used for the loss (Byzantine flips).
    Returns (model, info)."""
    c = {**default_cfg(), **cfg}
    dev = X.device
    yl = np.asarray(y if train_y is None else train_y)
    opt = torch.optim.SGD(model.parameters(), lr=c["lr"], momentum=c["momentum"], weight_decay=c["weight_decay"])
    gparams = [p.detach().clone() for p in global_model.parameters()] if (global_model is not None and c["mu_prox"] > 0) else None
    use_moon = c["mu_moon"] > 0 and global_model is not None and prev_model is not None
    P = torch.as_tensor(protos, dtype=torch.float32, device=dev) if (protos is not None and c["lam_proto"] > 0) else None
    yt = torch.as_tensor(yl, device=dev)
    losses, steps = [], 0
    model.train()
    # stratified batches are benign-first (see batches): pass nb so losses slice instead of boolean-indexing (no
    # device→host sync; same rows). 'random' batches are unordered → boolean masks.
    ordered = c["attack_frac"] != "random"
    meth = c.get("loss", "feco")
    cen = torch.as_tensor(c["center"], dtype=torch.float32, device=dev) if meth in ("svdd", "sad") else None
    for ep in range(c["epochs"]):
        # materialise the epoch's batches (consumes rng exactly as lazy iteration would) and upload indices once
        blist = list(batches(yl, rng, c["batch_size"], c["attack_frac"]))
        if not blist:
            continue
        all_idx = torch.as_tensor(np.concatenate([b for b, _ in blist]), device=dev)
        off = 0
        for idx, nbb in blist:
            ti = all_idx[off:off + len(idx)]
            off += len(idx)
            nb = nbb if ordered else None
            xb, yb = X[ti], yt[ti]
            z = model(xb)
            if meth == "clad":
                loss = clad_loss(z, yb, nb=nb)
            elif meth in ("svdd", "sad"):
                if meth == "svdd":  # unsupervised: benign only
                    if nb is not None:
                        loss = svdd_loss(z[:nb], yb[:nb], cen, eta=0.0, nb=nb)
                    else:
                        keep = yb == 0
                        loss = svdd_loss(z[keep], yb[keep], cen, eta=0.0)
                else:
                    loss = svdd_loss(z, yb, cen, eta=c.get("eta", 1.0), nb=nb)
            elif meth == "ae":
                if nb is not None:
                    loss = ((z[:nb] - xb[:nb]) ** 2).sum(1).mean()
                else:
                    keep = yb == 0
                    loss = ((z[keep] - xb[keep]) ** 2).sum(1).mean()
            else:
                loss = feco_loss(z, yb, tau=c["tau"], normalize=c["normalize"], nb=nb)
            if P is not None:
                loss = loss + c["lam_proto"] * proto_loss(z, yb, P, nb=nb)
            if gparams is not None:
                loss = loss + 0.5 * c["mu_prox"] * sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), gparams))
            if use_moon:
                with torch.no_grad():
                    zg = F.normalize(global_model(xb), dim=1)
                    zp = F.normalize(prev_model(xb), dim=1)
                zn = F.normalize(z, dim=1)
                pos = (zn * zg).sum(1) / c["tau_moon"]
                neg = (zn * zp).sum(1) / c["tau_moon"]
                loss = loss + c["mu_moon"] * (torch.logaddexp(pos, neg) - pos).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if c["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
            opt.step()
            losses.append(loss.detach())  # no per-step host sync (GPU); same values as .item()
            steps += 1
        if epoch_cb is not None and c["lam_proto"] > 0:
            P = torch.as_tensor(epoch_cb(model), dtype=torch.float32, device=dev)
    lv = torch.stack(losses).cpu().double().numpy() if losses else np.zeros(0)
    return model, {"loss": float(np.mean(lv)) if len(lv) else float("nan"), "steps": steps,
                   "attack_frac": c["attack_frac"] if (yl == 1).any() else 0.0}


def clone(model):
    return copy.deepcopy(model)
