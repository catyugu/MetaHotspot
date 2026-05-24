"""MetaHotspot Regression Tests.

End-to-end tests covering the full simulation pipeline using example cases.
"""

from pathlib import Path

import pytest
from pytest import approx

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
    """Path to the example1.xml test case."""
    return Path("./examples/example1/example1.xml")


@pytest.fixture
def example1_config(example1_path):
    """Parsed example1.xml configuration."""
    return parse_xml(example1_path)


@pytest.fixture
def example1_result(example1_path):
    """Full simulation result for example1.xml."""
    config = parse_xml(example1_path)
    mesh_topo = generate_mesh(config)
    fields, parsed_bcs = bake_model(config, mesh_topo)
    sys_mat = assemble_system(mesh_topo, fields, parsed_bcs, config)
    result = solve_system(sys_mat, mesh_topo, fields, config, parsed_bcs)
    return result


# ============================================================================
# Full Pipeline Regression Tests
# ============================================================================


class TestFullPipeline:
    """Test complete simulation pipeline for regression detection."""

    def test_example1_runs_successfully(self, example1_path):
        """Example1.xml completes without raising exceptions."""
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
        assert max_t > 656.0
        assert max_t < 660.0

    def test_example1_min_temperature_reasonable(self, example1_result):
        min_t = example1_result.min_temperature
        # Dirichlet boundary is set to 500K
        assert 499.0 < min_t < 501.0

    def test_example1_result_has_valid_indices(self, example1_result):
        """Result indices are valid and within temperature array bounds."""
        max_idx = example1_result.max_index
        min_idx = example1_result.min_index
        n = len(example1_result.temperatures)

        assert 0 <= max_idx < n
        assert 0 <= min_idx < n

    def test_example1_temperatures_shape_consistent(self, example1_config, example1_result):
        """Temperature array shape matches mesh topology."""
        config = example1_config
        mesh_topo = generate_mesh(config)

        expected_size = mesh_topo.n_x * mesh_topo.n_y * mesh_topo.n_z
        assert len(example1_result.temperatures) == expected_size


# ============================================================================
# Steady-State Tests
# ============================================================================


class TestSteadyState:
    """Tests specific to steady-state simulation."""

    def test_example1_is_steady_state(self, example1_config):
        """Example1.xml is configured as steady-state study."""
        assert example1_config.study_type == "Steady"

    def test_example1_time_series_empty(self, example1_result):
        """Steady-state result has empty time series."""
        assert len(example1_result.time_series) == 0


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])