"""Single re-run of NBA_clean_fedavg_s2 with the extended post-hoc list."""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location("p5", os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase5_grid.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


def configs():
    return [c for c in _m.configs() if c["run_id"] == "NBA_clean_fedavg_s2"]
