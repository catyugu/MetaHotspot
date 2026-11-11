"""Exercise the installed native solver alongside NumPy/SciPy in both import orders."""

import ctypes
import importlib.util
import os
import sys
from pathlib import Path

if sys.argv[1] == "native-first":
    # Importing the public package already imports NumPy. Load the installed
    # C API directly first so this really tests the opposite native load order.
    package = Path(importlib.util.find_spec("metahotspot").origin).parent
    name = {
        "linux": "libmhs_c_api.so",
        "darwin": "libmhs_c_api.dylib",
        "win32": "mhs_c_api.dll",
    }[sys.platform]
    if sys.platform == "win32":
        vendor = package.parent / "metahotspot.libs"
        if vendor.is_dir():
            dll_path = os.add_dll_directory(str(vendor))
    native = ctypes.CDLL(str(package / name))
    assert "numpy" not in sys.modules
    import numpy as np
    import scipy.linalg

    import metahotspot
else:
    import numpy as np
    import scipy.linalg

    scipy.linalg.solve(np.eye(8), np.ones(8))
    import metahotspot

from metahotspot import SolveOptions

# 15,625 cells select multiple OpenMP threads in the AMG solver. A constant
# source with fixed-temperature boundaries gives a finite, positive rise.
model = metahotspot.Model()
model.set_conformal_mesh(*(np.linspace(0, 0.01, 26) for _ in range(3)))
material = model.add_material("solid", "1", "1", "1", "1000", "500")
source = model.add_source("1e6")
volume = model.add_volume(material, source)
model.add_box(volume, [0, 0, 0], [0.01, 0.01, 0.01])
model.set_default_dirichlet("300")
solution = model.compile().solve(opts=SolveOptions(linear_solver="Amg"))
assert solution.state.size == 15625, solution.state.size
assert np.isfinite(solution.state).all()
assert solution.state.max() > 300.0
print(f"{sys.argv[1]}: 15,625-cell AMG solve, Tmax={solution.state.max():.6f}")
