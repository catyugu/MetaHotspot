import numpy as np

from metahotspot import Model


def test_octree_refinement_uses_public_model_contract():
    model = Model()
    model.set_cartesian_mesh([0, 1, 2], [0, 1], [0, 1])
    material = model.add_material("solid", "1", "1", "1", "1", "1")
    volume = model.add_volume(material)
    model.add_box(volume, [0, 0, 0], [2, 1, 1])
    model.refine_cartesian_box([0, 0, 0], [1, 1, 1], levels=1)
    compiled = model.compile()
    assert compiled.cell_count == 9
    assert compiled.info.state_count == 9
    assert np.isclose(compiled.fields.cell_volume.sum(), 2.0)
