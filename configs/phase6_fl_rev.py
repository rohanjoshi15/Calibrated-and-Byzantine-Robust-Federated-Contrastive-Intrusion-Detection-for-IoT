"""The 'fl' configs of phase6_grid.py in reverse order (second sweep from the other end)."""
import importlib.util
import os

os.environ["STAGE"] = "fl"
_spec = importlib.util.spec_from_file_location("p6", os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase6_grid.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


def configs():
    return list(reversed(_m.fl()))
