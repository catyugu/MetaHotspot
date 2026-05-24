"""MetaHotspot Regression Tests for example2.

End-to-end tests covering the full simulation pipeline using example2 case.
"""

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
def example2_path() -> Path:
    """Path to the example2.xml test case."""
    return Path("./examples/example2/example2.xml")


@pytest.fixture
def example2_config(example2_path):
    """Parsed example2.xml configuration."""
    return parse_xml(example2_path)


@pytest.fixture
def example2_result(example2_path):
    """Full simulation result for example2.xml."""
    config = parse_xml(example2_path)
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

    def test_example2_runs_successfully(self, example2_path):
        """Example2.xml completes without raising exceptions."""
        config = parse_xml(example2_path)
        mesh_topo = generate_mesh(config)
        fields, parsed_bcs = bake_model(config, mesh_topo)
        sys_mat = assemble_system(mesh_topo, fields, parsed_bcs, config)
        result = solve_system(sys_mat, mesh_topo, fields, config, parsed_bcs)

        assert result is not None
        assert hasattr(result, "max_temperature")
        assert hasattr(result, "min_temperature")

    def test_example2_max_temperature_reasonable(self, example2_result):
        """Max temperature is within expected range for steady-state with convection BCs."""
        max_t = example2_result.max_temperature
        assert 355.0 < max_t < 365.0

    def test_example2_min_temperature_reasonable(self, example2_result):
        """Min temperature is within expected range for steady-state with convection BCs."""
        min_t = example2_result.min_temperature
        # Convection boundary with EnvTemp=300K
        assert 322.0 < min_t < 325.0

    def test_example2_result_has_valid_indices(self, example2_result):
        """Result indices are valid and within temperature array bounds."""
        max_idx = example2_result.max_index
        min_idx = example2_result.min_index
        n = len(example2_result.temperatures)

        assert 0 <= max_idx < n
        assert 0 <= min_idx < n

    def test_example2_temperatures_shape_consistent(self, example2_config, example2_result):
        """Temperature array shape matches mesh topology."""
        config = example2_config
        mesh_topo = generate_mesh(config)

        expected_size = mesh_topo.n_x * mesh_topo.n_y * mesh_topo.n_z
        assert len(example2_result.temperatures) == expected_size


# ============================================================================
# Steady-State Tests
# ============================================================================


class TestSteadyState:
    """Tests specific to steady-state simulation."""

    def test_example2_is_steady_state(self, example2_config):
        """Example2.xml is configured as steady-state study."""
        assert example2_config.study_type == "Steady"

    def test_example2_time_series_empty(self, example2_result):
        """Steady-state result has empty time series."""
        assert len(example2_result.time_series) == 0


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])