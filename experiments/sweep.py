"""Run many configs in parallel worker processes (1 torch thread each). Skips finished runs.

python experiments/sweep.py <grid_module> [--workers N]
<grid_module>: python file in configs/ defining `configs()` -> list of config dicts (each with phase, run_id)."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS
import argparse
import subprocess
import importlib.util
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _work(cfg):
    import torch
    torch.set_num_threads(1)
    from experiments.run import execute
    t = time.time()
    st = execute(cfg)
    return cfg["run_id"], st, time.time() - t


def load_grid(path):
    spec = importlib.util.spec_from_file_location("grid", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.configs()


def mem_available_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def gpu_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa
        return 0


def gpu_pids():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
        return {int(x) for x in out.split() if x.strip().isdigit()}
    except Exception:  # noqa
        return set()


def main(path, workers, task_timeout=4 * 3600, min_free_mb=3000, start_gap_s=30, device=None, vram_cap_mb=7000,
         vram_per_task_mb=900, max_gpu_procs=6):
    """Dispatches at most `workers` tasks at once, and starts a new task only if MemAvailable ≥ min_free_mb,
    ≥ start_gap_s since the previous start (lets the new task reach its memory peak before the next decision) and,
    for cuda grids, GPU memory used + vram_per_task_mb ≤ vram_cap_mb and the number of GPU processes (this sweep's
    tasks + CUDA processes of other sweeps) stays < max_gpu_procs. device: if given, sets cfg['device'] where the
    config has none."""
    cfgs = load_grid(path)
    if device:
        for c in cfgs:
            c.setdefault("device", device)
    use_gpu = any(c.get("device") == "cuda" for c in cfgs)
    todo = [c for c in cfgs if not os.path.exists(os.path.join("results", c["phase"], c["run_id"], "metrics.json"))]
    print(f"{len(cfgs)} configs, {len(todo)} to run, ≤{workers} workers, min free {min_free_mb} MB, gpu={use_gpu}", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:  # fresh process per task: frees cached data (OOM safety)
        queue = list(todo)
        running = {}
        done, last_start = 0, 0.0
        while queue or running:
            time.sleep(5)
            for rid in list(running):
                ar, ts, ph = running[rid]
                if ar.ready():
                    try:
                        _, st, dt = ar.get()
                    except Exception as e:  # noqa
                        st, dt = f"error:{e!r}", time.time() - ts
                    del running[rid]
                    done += 1
                    el = time.time() - t0
                    print(f"[{done}/{len(todo)}] {st} {rid} {dt:.0f}s | elapsed {el / 60:.1f}m "
                          f"eta {(el / done) * (len(todo) - done) / 60:.1f}m | free {mem_available_mb()} MB"
                          f"{f' | vram {gpu_used_mb()} MB' if use_gpu else ''}", flush=True)
                elif time.time() - ts > task_timeout:
                    del running[rid]
                    done += 1
                    print(f"[{done}/{len(todo)}] timeout {rid} (worker lost?)", flush=True)
            if queue and len(running) < workers and time.time() - last_start >= start_gap_s \
                    and mem_available_mb() >= min_free_mb \
                    and (not use_gpu or (gpu_used_mb() + vram_per_task_mb <= vram_cap_mb
                                         and len(running) + len(gpu_pids() - {p.pid for p in pool._pool}) < max_gpu_procs)):
                c = queue.pop(0)
                running[c["run_id"]] = (pool.apply_async(_work, (c,)), time.time(), c["phase"])
                last_start = time.time()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-free-mb", type=int, default=3000)
    ap.add_argument("--device", default=None, help="cpu|cuda: default device for configs without one")
    a = ap.parse_args()
    main(a.grid, a.workers, min_free_mb=a.min_free_mb, device=a.device)
