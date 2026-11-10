import json
import os
import random
import time

import numpy as np
import torch
import yaml


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available() and DEV.type == "cuda":
        torch.cuda.manual_seed_all(s)


# Process-wide compute device. Set once per run by src.pipeline.run from cfg["device"].
DEV = torch.device("cpu")


def set_device(name="cpu"):
    """Select the torch device for this process. For cuda, enable deterministic kernels:
    CUBLAS_WORKSPACE_CONFIG=:4096:8, use_deterministic_algorithms(warn_only), cudnn deterministic, no benchmark."""
    global DEV
    if name == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("cfg device=cuda but torch.cuda.is_available() is False")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    DEV = torch.device(name)
    return DEV


def device_info():
    d = {"device": DEV.type, "torch": torch.__version__}
    if DEV.type == "cuda":
        d["gpu"] = torch.cuda.get_device_name(0)
        d["cuda"] = torch.version.cuda
    return d


def device():
    return DEV


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    if isinstance(o, float) and (o != o):
        return None
    return o


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(to_py(obj), f, indent=1)
    os.replace(tmp, path)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def dump_yaml(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(to_py(obj), f, sort_keys=False)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return time.time() - self.t0
