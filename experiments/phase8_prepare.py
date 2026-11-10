"""Before a GPU re-run sweep: move every earlier run dir that the stage will re-run to
_trash/phase8_superseded/<phase>/<run_id>. Only dirs without the metrics.json key 'device' (earlier CPU runs) or
without metrics.json (failed / partial runs) are moved, so the script is safe to re-run. Refuses if no
results/_v1 copy exists for a CPU run.
STAGE=<stages> python experiments/phase8_prepare.py [--dry]"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import load_grid  # noqa: E402


def main(dry=False):
    cfgs = load_grid("configs/phase8_grid.py")
    moved = kept = new = 0
    for c in cfgs:
        d = os.path.join("results", c["phase"], c["run_id"])
        if not os.path.isdir(d):
            new += 1
            continue
        mp = os.path.join(d, "metrics.json")
        if os.path.exists(mp) and "device" in json.load(open(mp)):
            kept += 1
            continue
        if os.path.exists(mp):
            v1 = os.path.join("results", "_v1", c["phase"], c["run_id"], "metrics.json")
            assert os.path.exists(v1), f"no v1 archive for {d}"
        dst = os.path.join("_trash", "phase8_superseded", c["phase"], c["run_id"])
        if os.path.exists(dst):
            dst += f"_{int(os.path.getmtime(d))}"
        if not dry:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(d, dst)
        moved += 1
    print(f"{len(cfgs)} configs: moved {moved} v1/partial dirs to _trash/phase8_superseded, kept {kept} phase-8 runs, "
          f"{new} new", flush=True)


if __name__ == "__main__":
    main("--dry" in sys.argv)
