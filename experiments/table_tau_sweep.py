"""T_tau_sweep_val: central FeCo tuning grid (NSL-KDD, seed 0), VALIDATION metrics only.
The tuning runs are CPU runs that were not re-run; their test fields used the mirror test file and are never read.
Output: results/tables/T_tau_sweep_val.csv.   python experiments/table_tau_sweep.py"""
import glob
import json
import os

import pandas as pd
import yaml

rows = []
for d in sorted(glob.glob("results/phase1/tuneC_*")):
    mp = os.path.join(d, "metrics.json")
    if not os.path.exists(mp):
        continue
    m, c = json.load(open(mp)), yaml.safe_load(open(os.path.join(d, "config.yaml")))
    t, v = c["train"], m["results"]["template"]["val"]
    rows.append({"run_id": os.path.basename(d), "tau": t["tau"], "momentum": t["momentum"], "batch": t["batch_size"],
                 "epochs": t["epochs"], "normalize": t.get("normalize", True), "val_tpr1": v["ranking"]["tpr@1"],
                 "val_tpr5": v["ranking"]["tpr@5"], "val_auroc": v["ranking"]["auroc"], "val_auprc": v["ranking"]["auprc"],
                 "val_acc_p5": v["feco_p5"]["acc"], "val_fpr_p5": v["feco_p5"]["fpr"],
                 "train_loss": m["train_info"]["loss"],
                 "device": m.get("device", "cpu")})
df = pd.DataFrame(rows)
df.to_csv("results/tables/T_tau_sweep_val.csv", index=False, float_format="%.4f")
print(df.to_string())
