import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """FeCo encoder: d -> 128 -> 256 -> out (ReLU). FeCo TDSC §IV-C2."""

    def __init__(self, d, hidden=(128, 256), out=128, bias=True):
        super().__init__()
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h, bias=bias), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AE(nn.Module):
    """Autoencoder baseline: encoder = MLP, mirrored decoder."""

    def __init__(self, d, hidden=(128, 256), out=128):
        super().__init__()
        self.enc = MLP(d, hidden, out)
        layers, prev = [nn.ReLU()], out
        for h in reversed(hidden):
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, d))
        self.dec = nn.Sequential(*layers)

    def forward(self, x):
        return self.dec(self.enc(x))


def model_device(model):
    p = next(model.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


@torch.no_grad()
def embed(model, X, normalize=True, bs=8192):
    """X: numpy array or tensor (any device). Runs on the model's device; returns a CPU float32 numpy array."""
    model.eval()
    dev = model_device(model)
    out = []
    Xt = torch.as_tensor(X)
    for i in range(0, len(Xt), bs):
        z = model(Xt[i:i + bs].to(dev, non_blocking=False))
        out.append((F.normalize(z, dim=1) if normalize else z).cpu())
    model.train()
    return torch.cat(out).numpy() if out else torch.zeros(0, 1).numpy()
