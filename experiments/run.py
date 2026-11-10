"""Run one experiment from a YAML config (or dict). Skips if metrics.json exists.

python experiments/run.py configs/<file>.yaml [--force]
Writes results/<phase>/<run_id>/{config.yaml, metrics.json, stdout.log}."""
import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; harmless on CPU
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.common import dump_json, dump_yaml, load_yaml  # noqa: E402


def run_dir(cfg):
    return os.path.join("results", cfg.get("phase", "misc"), cfg["run_id"])


def execute(cfg, force=False):
    from src.pipeline import run
    d = run_dir(cfg)
    mpath = os.path.join(d, "metrics.json")
    if os.path.exists(mpath) and not force:
        return "skip"
    os.makedirs(d, exist_ok=True)
    dump_yaml(cfg, os.path.join(d, "config.yaml"))
    logf = open(os.path.join(d, "stdout.log"), "w")

    def log(msg):
        logf.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        logf.flush()

    t0 = time.time()
    try:
        out = run(cfg, log=log)
        out["runtime_s"] = time.time() - t0
        dump_json(out, mpath)
        log(f"done in {out['runtime_s']:.1f}s")
        return "ok"
    except Exception:  # noqa
        log(traceback.format_exc())
        return "fail"
    finally:
        logf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(execute(load_yaml(a.config), a.force))
