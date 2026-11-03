"""Exercise the installed native solver alongside NumPy/SciPy in both import orders."""

import sys

if sys.argv[1] == "native-first":
    import metahotspot
    from metahotspot._lib import get_dll

    get_dll()
    import numpy as np
    import scipy.linalg
else:
    import numpy as np
    import scipy.linalg

    scipy.linalg.solve(np.eye(8), np.ones(8))
    import metahotspot

from metahotspot import SolveOptions
from metahotspot.enums import GeometryOp, LengthUnit

# 15,625 cells select multiple OpenMP threads in the AMG solver. A constant
# source with fixed-temperature boundaries gives a finite, positive rise.
model = metahotspot.Model()
model.set_settings(length_unit=LengthUnit.METER)
model.set_mesh(*(np.linspace(0, 0.01, 26) for _ in range(3)))
model.add_material("solid", "1", "1", "1", "1000", "500")
layer = model.add_layer("0.01")
block = model.add_block(layer, "solid", heat_source="1e6")
model.add_rect(block, GeometryOp.ADD, "0", "0", "0.01", "0.01")
model.set_default_dirichlet("300")
solution = model.compile().solve(opts=SolveOptions(linear_solver="AmgCg"))
assert solution.state.size == 15625, solution.state.size
assert np.isfinite(solution.state).all()
assert solution.state.max() > 300.0
print(f"{sys.argv[1]}: 15,625-cell AMG solve, Tmax={solution.state.max():.6f}")
