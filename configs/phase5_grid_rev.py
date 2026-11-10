"""Same configs as phase5_grid.py in reverse order (second sweep working from the other end)."""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location("p5", os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase5_grid.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


def configs():
    return list(reversed(_m.configs()))
