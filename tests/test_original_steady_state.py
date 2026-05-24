"""MetaHotspot Regression Tests for original steady-state cases.

End-to-end tests covering the full simulation pipeline using example1 and example2 cases.
"""

from pathlib import Path

import pytest

from metahotspot.xml_parser import parse_xml
from metahotspot.mesher import generate_mesh
from metahotspot.baker import bake_model
from metahotspot.assembler import assemble_system
from metahotspot.solver import solve_system


@pytest.fixture
def case1_result(case1_path):
    config = parse_xml(case1_path)
    mesh_topo = generate_mesh(config)
    fields, parsed_bcs = bake_model(config, mesh_topo)
    sys_mat = assemble_system(mesh_topo, fields, parsed_bcs, config)
    return solve_system(sys_mat, mesh_topo, fields, config, parsed_bcs)


@pytest.fixture
def case2_result(case2_path, case2_vtu_path):
    config = parse_xml(case2_path)
    mesh_topo = generate_mesh(config)
    fields, parsed_bcs = bake_model(config, mesh_topo)
    sys_mat = assemble_system(mesh_topo, fields, parsed_bcs, config)
    return solve_system(sys_mat, mesh_topo, fields, config, parsed_bcs, output_vtu=str(case2_vtu_path))


@pytest.fixture
def case1_path():
    return Path("./cases/original_steady_tests/case1.xml")


@pytest.fixture
def case2_path():
    return Path("./cases/original_steady_tests/case2.xml")


@pytest.fixture
def case2_vtu_path():
    return Path("outputs/case2_result.vtu")


class TestExample1:
    def test_example1(self, case1_result):
        assert case1_result.max_temperature is not None
        assert case1_result.min_temperature is not None
        assert 656.0 < case1_result.max_temperature < 660.0
        assert 500.0 < case1_result.min_temperature < 503.0
        n = len(case1_result.temperatures)
        assert 0 <= case1_result.max_index < n
        assert 0 <= case1_result.min_index < n


class TestExample2:
    def test_example2(self, case2_result, case2_vtu_path):
        assert case2_result.max_temperature is not None
        assert case2_result.min_temperature is not None
        assert 355.0 < case2_result.max_temperature < 365.0
        assert 322.0 < case2_result.min_temperature < 325.0
        n = len(case2_result.temperatures)
        assert 0 <= case2_result.max_index < n
        assert 0 <= case2_result.min_index < n
        assert case2_vtu_path.exists()
        assert case2_vtu_path.stat().st_size > 0