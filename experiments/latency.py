"""CPU inference latency of the FeCo encoder + scoring: fp32 vs dynamic int8 (torch.ao.quantization.quantize_dynamic
on nn.Linear). Trains the val-selected central NSL-KDD config (corrected protocol, seed 0), then times scoring;
also reports test metrics with a conformal α=0.05 threshold and fp32/int8 decision agreement. Single thread,
batch sizes 1 and 1024.
Outputs results/tables/T6_latency.csv. python experiments/latency.py"""
import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.calib import thresholds as th  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.models import detect as det  # noqa: E402
from src.models.encoder import MLP, embed  # noqa: E402
from src.models.train import train_local  # noqa: E402
from src.pipeline import build_data  # noqa: E402
from src.utils import common as U  # noqa: E402


def time_it(fn, X, reps):
    fn(X)
    t = time.perf_counter()
    for _ in range(reps):
        fn(X)
    return (time.perf_counter() - t) / reps


def main():
    torch.set_num_threads(1)
    sel = json.load(open("configs/phase1_selected.json"))["central"]
    rows = []
    for ds, dcfg, tr in (("NSL", {"name": "nslkdd", "fs": "procedure"}, sel),):
        D = build_data(dcfg, 0)
        dev = U.set_device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)
        m = MLP(D["dim"]).to(dev)
        m, _ = train_local(m, torch.as_tensor(D["train"]["X"]).to(dev), D["train"]["y"], tr, np.random.RandomState(0))
        m = m.cpu()  # latency is measured on CPU (single thread), fp32 vs dynamic int8
        m.eval()
        q = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
        Zb = embed(m, D["train"]["X"][D["train"]["y"] == 0][:50000])
        zbar = det.template(Zb)
        res = {}
        for name, mdl in (("fp32", m), ("int8_dynamic", q)):
            Zbq = embed(mdl, D["train"]["X"][D["train"]["y"] == 0][:50000])
            zq = det.template(Zbq)
            s_cal = det.score_template(embed(mdl, D["calib"]["X"]), zq)
            rho = th.conformal(s_cal, 0.05)
            s_te = det.score_template(embed(mdl, D["test"]["X"]), zq)
            ev = M.evaluate(D["test"]["y"], s_te, rho)
            res[name] = (s_te < rho).astype(int)

            def f(X, mdl=mdl, zq=zq):
                with torch.no_grad():
                    return det.score_template(torch.nn.functional.normalize(mdl(X), dim=1).numpy(), zq)
            for bs, reps in ((1, 2000), (1024, 50)):
                X = torch.as_tensor(D["test"]["X"][:bs])
                dt = time_it(f, X, reps)
                rows.append({"dataset": ds, "model": name, "batch": bs, "train_device": dev.type, "infer_device": "cpu",
                             "ms_per_batch": 1000 * dt,
                             "us_per_flow": 1e6 * dt / bs, "test_acc@conf0.05": 100 * ev["acc"],
                             "test_fpr@conf0.05": 100 * ev["fpr"], "auroc": 100 * ev["auroc"]})
        size = {"fp32": sum(p.numel() * 4 for p in m.parameters())}
        rows.append({"dataset": ds, "model": "agreement_int8_vs_fp32", "batch": None,
                     "ms_per_batch": None, "us_per_flow": None,
                     "test_acc@conf0.05": 100 * float(np.mean(res["fp32"] == res["int8_dynamic"])),
                     "test_fpr@conf0.05": None, "auroc": None, "fp32_param_bytes": size["fp32"]})
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/T6_latency.csv", index=False, float_format="%.4f")
    print(df.to_string())


if __name__ == "__main__":
    main()
