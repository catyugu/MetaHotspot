"""Native topology, geometry and algebraic ordering are separate contracts."""

import numpy as np
from test_model_binding import _model


def test_compiled_snapshot_exposes_explicit_faces_and_cell_bounds():
    compiled = _model().compile()
    assert compiled.metadata.face_count == 6
    assert compiled.metadata.state_count == 1
    cells = compiled.cells
    assert cells.cell_min.shape == (1, 3)
    assert cells.cell_max.shape == (1, 3)
    np.testing.assert_allclose(cells.cell_max - cells.cell_min, 1e-3)
    assert cells.face_owner.shape == (6,)
    assert cells.face_min.shape == (6, 3)
    np.testing.assert_array_equal(cells.cell_to_dof, [0])
    assert not hasattr(cells, "grid_to_cell")
    assert not hasattr(cells, "cell_to_grid")
