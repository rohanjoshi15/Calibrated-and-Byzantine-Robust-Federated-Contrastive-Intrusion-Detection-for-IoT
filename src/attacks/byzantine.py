"""Byzantine clients (C3).

attack:
  'whitelist'  model poisoning: the attacker trains on its own data with its botnet flows relabelled benign
               (train_y = 0 → the FeCo loss pulls them into the benign cluster) and scales its update by `boost`
               (v = g + boost·(v − g)).
  'proto'      prototype poisoning: attacker reports centroids of its *attack* embeddings (fake benign prototypes in
               the attack region) with counts inflated by `boost` × its benign count.
  'calib'      calibration poisoning: attacker reports a fabricated score histogram concentrated at `calib_value`
               (default −1, i.e. 'benign looks very anomalous') with counts × `boost`, dragging the global ρ down.
  'none'       clients flagged but honest (control).
Attack success rate (ASR) = fraction of the attackers' own botnet flows (their training attack rows) classified benign
by the final global model + prototypes under the global threshold (conformal_global α=0.05 by default) and under the
honest per-client conformal threshold."""
import numpy as np
import torch

from ..calib import thresholds as th
from ..calib.robust import robust_quantile
from ..models import detect as det
from ..models.encoder import embed
from ..models.train import train_local


class Byzantine:
    def __init__(self, cfg, D, C, seed):
        self.cfg = cfg
        self.attack = cfg.get("attack", "whitelist")
        self.boost = float(cfg.get("boost", 1.0))
        rng = np.random.RandomState(seed + 4242)
        n = len(C["train"])
        y = D["train"]["y"]
        has_att = [i for i, p in enumerate(C["train"]) if (y[p] == 1).sum() > 0]
        k = int(round(cfg.get("frac", 0.2) * n))
        pool = has_att if len(has_att) >= k else list(range(n))
        self.ids = set(int(i) for i in rng.choice(pool, k, replace=False))
        self.C, self.D = C, D
        self.att_rows = {i: C["train"][i][y[C["train"][i]] == 1] for i in self.ids}

    def local(self, cid, model, Xc, yc, tcfg, rng, global_model):
        if self.attack == "whitelist":
            # Relabelling all flows benign leaves the FeCo loss without negatives (≡0), so the operative mechanism
            # is the flipped objective: pull own attack embeddings onto the benign mean while keeping benign compact.
            model = self._pull(model, Xc, yc, tcfg, rng)
        else:
            model, _ = train_local(model, Xc, yc, tcfg, rng, global_model=global_model)
        return model

    def _pull(self, model, Xc, yc, tcfg, rng):
        """Flipped objective: maximise cos(u, v̄) for own attack embeddings u (v̄ = benign mean, detached)."""
        ia, ib = np.flatnonzero(yc == 1), np.flatnonzero(yc == 0)
        if len(ia) == 0 or len(ib) == 0:
            return model
        lr = tcfg.get("lr", 1e-3) * self.cfg.get("atk_lr_mult", 10.0)  # attacker controls its own optimiser
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=tcfg.get("momentum", 0.9))
        for _ in range(tcfg.get("epochs", 4) * max(1, int(np.ceil(len(ib) / 256)))):
            a = torch.as_tensor(rng.choice(ia, min(64, len(ia))), device=Xc.device)
            b = torch.as_tensor(rng.choice(ib, min(256, len(ib))), device=Xc.device)
            za = torch.nn.functional.normalize(model(Xc[a]), dim=1)
            zb = torch.nn.functional.normalize(model(Xc[b]), dim=1)
            loss = -(za @ zb.mean(0).detach()).mean() - (zb @ zb.mean(0).detach()).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return model

    def post(self, cid, v, g):
        if self.attack == "whitelist" and self.boost != 1.0:
            return g + self.boost * (v - g)
        return v

    def set_model(self, model):
        self._model = model

    def proto_poison(self, cid, C, cnt):
        if self.attack != "proto" or cid not in self.ids or len(self.att_rows[cid]) == 0:
            return C, cnt
        Za = embed(self._model, self.D["train"]["X"][self.att_rows[cid]], normalize=True)
        K = max(len(C), 1)
        if K == 1:
            Cf, cf = det.unit(Za.mean(0, keepdims=True)), np.array([len(Za)], float)
        else:
            Cf, cf = det.spherical_kmeans(Za, min(K, len(Za)), seed=cid, n_init=1, iters=20)
        tot = max(cnt.sum(), 1.0)
        return Cf, cf / max(cf.sum(), 1) * tot * self.boost

    def robust_protos(self, Cs, cnts, K, how):
        """Count-capped (at median client total) weighted k-means, then each prototype = robust aggregate
        (geometric median or coordinate trimmed mean) of the client centroids assigned to it."""
        tot = np.array([c.sum() for c in cnts])
        cap = np.median(tot[tot > 0])
        cnts = [c * min(1.0, cap / max(c.sum(), 1e-9)) for c in cnts]
        C = np.concatenate([c for c in Cs if len(c)])
        w = np.concatenate([c for c in cnts if len(c)])
        owner = np.concatenate([np.full(len(c), i) for i, c in enumerate(Cs) if len(c)])
        if K == 1:
            groups = [np.arange(len(C))]
        else:
            P0, _ = det.spherical_kmeans(C, min(K, len(C)), w=w, seed=0, n_init=3)
            a = np.argmax(C @ P0.T, 1)
            groups = [np.flatnonzero(a == k) for k in range(len(P0))]
        if how == "support":
            # keep a prototype only if more distinct clients back it than the assumed max number of attackers
            n_cli = len([c for c in Cs if len(c)])
            s_min = int(np.ceil(self.cfg.get("f_max", 0.3) * n_cli)) + 1
            kept = [g for g in groups if len(np.unique(owner[g])) >= s_min]
            self.last_support = {"s_min": s_min, "kept": len(kept), "total": len(groups)}
            groups = kept if kept else [max(groups, key=lambda g: len(np.unique(owner[g])))]
        out = []
        for g in groups:
            if len(g) == 0:
                continue
            Cg = torch.as_tensor(C[g])
            if how in ("geomed", "support"):
                from ..fl.aggregators import geometric_median
                out.append(geometric_median(Cg).numpy())
            else:
                from ..fl.aggregators import trimmed_mean
                out.append(trimmed_mean(Cg, beta=self.cfg.get("trim_beta", 0.2)).numpy())
        return det.unit(np.array(out))

    def calib_poison(self, cid, s):
        if self.attack != "calib" or cid not in self.ids:
            return s
        n = int(max(len(s), 1) * self.boost)
        return np.full(n, self.cfg.get("calib_value", -1.0))

    def evaluate(self, model, P, D, C, alpha=0.05):
        self._model = model
        sf = lambda X: det.score_maxcos(embed(model, X, normalize=True), P)  # noqa
        X = D["train"]["X"]
        s_cal = [sf(D["calib"]["X"][p]) if len(p) else np.zeros(0) for p in C["calib_benign"]]
        honest = [i for i in range(len(s_cal)) if i not in self.ids]
        rep = [self.calib_poison(i, s) for i, s in enumerate(s_cal)]
        hists = [th.histogram(s) for s in rep]
        rho_g = th.from_histogram(sum(hists), alpha)
        rho_r, keep, _ = robust_quantile(hists, alpha, f_max=self.cfg.get("f_max", 0.3))
        rho_honest = th.from_histogram(sum(hists[i] for i in honest), alpha)
        out = {"rho_global": rho_g, "rho_robust": rho_r, "rho_honest_oracle": rho_honest,
               "robust_kept_byz": int(len(set(keep.tolist()) & self.ids)), "n_byz": len(self.ids)}
        rows = np.concatenate([r for r in self.att_rows.values()]) if self.att_rows else np.array([], int)
        if len(rows):
            s_att = sf(X[rows])
            for nm, r in (("global", rho_g), ("robust", rho_r), ("honest_oracle", rho_honest)):
                out[f"asr_{nm}"] = float(np.mean(s_att >= r))
        return out


def make_byzantine(cfg, D, C, seed):
    b = Byzantine(cfg, D, C, seed)
    return b
