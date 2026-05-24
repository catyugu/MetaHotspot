"""MetaHotspot Regression Tests for original steady-state cases.

End-to-end tests covering the full simulation pipeline using example1 and example2 cases.
"""

import os
from pathlib import Path

import pytest

from metahotspot.xml_parser import parse_xml
from metahotspot.mesher import generate_mesh
from metahotspot.baker import bake_model
from metahotspot.assembler import assemble_system
from metahotspot.solver import solve_system


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def example1_path() -> Path:
    return Path("./cases/original_steady_tests/case1.xml")


@pytest.fixture
def example2_path() -> Path:
    return Path("./cases/original_steady_tests/case2.xml")


@pytest.fixture
def example1_config(example1_path):
    return parse_xml(example1_path)


@pytest.fixture
def example2_config(example2_path):
    return parse_xml(example2_path)


@pytest.fixture
def example1_result(example1_path):
    config = parse_xml(example1_path)
    mesh_topo = generate_mesh(config)
    fields, parsed_bcs = bake_model(config, mesh_topo)
    sys_mat = assemble_system(mesh_topo, fields, parsed_bcs, config)
    result = solve_system(sys_mat, mesh_topo, fields, config, parsed_bcs)
    return result


@pytest.fixture
def example2_result(example2_path):
    config = parse_xml(example2_path)
    mesh_topo = generate_mesh(config)
    fields, parsed_bcs = bake_model(config, mesh_topo)
    sys_mat = assemble_system(mesh_topo, fields, parsed_bcs, config)
    result = solve_system(sys_mat, mesh_topo, fields, config, parsed_bcs)
    return result


# ============================================================================
# Example1 Tests
# ============================================================================


class TestExample1:
    def test_example1_runs_successfully(self, example1_path):
        config = parse_xml(example1_path)
        mesh_topo = generate_mesh(config)
        fields, parsed_bcs = bake_model(config, mesh_topo)
        sys_mat = assemble_system(mesh_topo, fields, parsed_bcs, config)
        result = solve_system(sys_mat, mesh_topo, fields, config, parsed_bcs)
        assert result is not None
        assert hasattr(result, "max_temperature")
        assert hasattr(result, "min_temperature")

    def test_example1_max_temperature_reasonable(self, example1_result):
        max_t = example1_result.max_temperature
        assert 656.0 < max_t < 660.0

    def test_example1_min_temperature_reasonable(self, example1_result):
        min_t = example1_result.min_temperature
        assert 500.0 < min_t < 503.0

    def test_example1_result_has_valid_indices(self, example1_result):
        max_idx = example1_result.max_index
        min_idx = example1_result.min_index
        n = len(example1_result.temperatures)
        assert 0 <= max_idx < n
        assert 0 <= min_idx < n

    def test_example1_temperatures_shape_consistent(self, example1_config, example1_result):
        config = example1_config
        mesh_topo = generate_mesh(config)
        expected_size = (mesh_topo.n_x - 1) * (mesh_topo.n_y - 1) * (mesh_topo.n_z - 1)
        assert len(example1_result.temperatures) == expected_size

    def test_example1_is_steady_state(self, example1_config):
        assert example1_config.study_type == "Steady"

    def test_example1_time_series_empty(self, example1_result):
        assert len(example1_result.time_series) == 0


# ============================================================================
# Example2 Tests
# ============================================================================


class TestExample2:
    def test_example2_runs_successfully(self, example2_path):
        config = parse_xml(example2_path)
        mesh_topo = generate_mesh(config)
        fields, parsed_bcs = bake_model(config, mesh_topo)
        sys_mat = assemble_system(mesh_topo, fields, parsed_bcs, config)
        result = solve_system(sys_mat, mesh_topo, fields, config, parsed_bcs)
        assert result is not None
        assert hasattr(result, "max_temperature")
        assert hasattr(result, "min_temperature")

    def test_example2_max_temperature_reasonable(self, example2_result):
        max_t = example2_result.max_temperature
        assert 355.0 < max_t < 365.0

    def test_example2_min_temperature_reasonable(self, example2_result):
        min_t = example2_result.min_temperature
        assert 322.0 < min_t < 325.0

    def test_example2_result_has_valid_indices(self, example2_result):
        max_idx = example2_result.max_index
        min_idx = example2_result.min_index
        n = len(example2_result.temperatures)
        assert 0 <= max_idx < n
        assert 0 <= min_idx < n

    def test_example2_temperatures_shape_consistent(self, example2_config, example2_result):
        config = example2_config
        mesh_topo = generate_mesh(config)
        expected_size = (mesh_topo.n_x - 1) * (mesh_topo.n_y - 1) * (mesh_topo.n_z - 1)
        assert len(example2_result.temperatures) == expected_size

    def test_example2_is_steady_state(self, example2_config):
        assert example2_config.study_type == "Steady"

    def test_example2_time_series_empty(self, example2_result):
        assert len(example2_result.time_series) == 0

    def test_example2_vtu_export(self, example2_path):
        config = parse_xml(example2_path)
        mesh_topo = generate_mesh(config)
        fields, parsed_bcs = bake_model(config, mesh_topo)
        sys_mat = assemble_system(mesh_topo, fields, parsed_bcs, config)
        vtu_path = Path("outputs/case2_result.vtu")
        if vtu_path.exists():
            vtu_path.unlink()
        result = solve_system(sys_mat, mesh_topo, fields, config, parsed_bcs, output_vtu=str(vtu_path))
        assert vtu_path.exists()
        assert vtu_path.stat().st_size > 0


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])