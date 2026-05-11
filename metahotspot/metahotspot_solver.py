import os
import meshio
import numpy as np
import time

from metahotspot.logging_config import get_logger
from metahotspot.assembler import FVMAssembler
from metahotspot.thermal_solver import ThermalSolver
from metahotspot.mesh_preprocessor import MeshPreprocessor
from metahotspot.fluid_preprocessor import FluidPreprocessor
from metahotspot.metahotspot_types import MeshTopology
from metahotspot.model25d import parse_computational_model
from metahotspot.numba_warmup import warmup_numba_kernels
from metahotspot.gmsh_mesher import GmshMesher

_logger = get_logger(__name__)


class MetaHotspotSolver:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.base_dir = os.path.dirname(config_path)

        (
            self.solver_config,
            self.layer_regions,
        ) = parse_computational_model(config_path)

        self.mesh: meshio.Mesh = None

    def run(self):
        warmup_start = time.perf_counter()
        warmup_numba_kernels()
        warmup_end = time.perf_counter()
        _logger.info(
            f"Numba kernels warmup completed in {warmup_end - warmup_start:.2f} seconds"
        )

        start = time.perf_counter()
        mesher = GmshMesher()
        self.mesh = mesher.generate_mesh(self.layer_regions)
        mesh_gen_finished = time.perf_counter()
        _logger.info(
            f"Mesh generation completed in {mesh_gen_finished - start:.2f} seconds"
        )

        topo, fields = MeshPreprocessor(
            self.solver_config.default_solid, self.layer_regions
        ).process(self.mesh)
        mesh_pre_finished = time.perf_counter()
        _logger.info(
            f"Mesh preprocessing completed in {mesh_pre_finished - mesh_gen_finished:.2f} seconds"
        )

        FluidPreprocessor(self.solver_config.boundary_conditions).solve_flow(
            topo, fields
        )
        pressure_solve_finished = time.perf_counter()
        _logger.info(
            f"Fluid flow solving completed in {pressure_solve_finished - mesh_pre_finished:.2f} seconds"
        )

        matrices = FVMAssembler(
            topo, fields, self.solver_config.boundary_conditions, self.layer_regions
        ).assemble()
        assembly_finished = time.perf_counter()
        _logger.info(
            f"System matrix assembly completed in {assembly_finished - pressure_solve_finished:.2f} seconds"
        )

        solver = ThermalSolver(matrices)
        ptrace = self._load_ptrace()

        if self.solver_config.simulation_type == "steady":
            temperatures = solver.solve_steady(
                np.array(
                    [
                        np.mean([s.get(n, 0.0) for s in ptrace])
                        for n in matrices.unit_names
                    ]
                )
                if ptrace
                else np.zeros(len(matrices.unit_names))
            )
            out_filename = "result.vtu"
        else:
            temperatures = solver.solve_transient(
                self.solver_config.timestep,
                ptrace,
                self._get_init_temp(topo),
                topo.volumes,
                fields.cp,
            )
            out_filename = "transient_result.vtu"

        end = time.perf_counter()
        _logger.info(
            f"Thermal solving completed in {end - assembly_finished:.2f} seconds"
        )
        _logger.info(f"Simulation completed in {end - start:.2f} seconds")
        _logger.info(f"Exporting results to {out_filename}...")
        self._export_vtu(topo, temperatures, out_filename, self.mesh)

    def _load_ptrace(self) -> list[dict]:
        if not self.solver_config.ptrace_file_path:
            return []
        path = os.path.join(self.base_dir, self.solver_config.ptrace_file_path)
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            headers = f.readline().split()
            return [dict(zip(headers, map(float, l.split()))) for l in f if l.strip()]

    def _get_init_temp(self, topo: MeshTopology) -> np.ndarray:
        temp = np.full(topo.n_cells, self.solver_config.init_temperature)
        init_file = self.solver_config.init_temperature_file_path
        if init_file and os.path.exists(os.path.join(self.base_dir, init_file)):
            init_mesh = meshio.read(os.path.join(self.base_dir, init_file))
            offset = 0
            for block, block_temps in zip(
                init_mesh.cells, init_mesh.cell_data.get("Temperature_K", [])
            ):
                if block.type == "hexahedron":
                    count = len(block_temps)
                    valid_ids = np.arange(offset, offset + count)
                    valid_mask = valid_ids < len(topo.orig_to_new_id)
                    temp[topo.orig_to_new_id[valid_ids[valid_mask]]] = block_temps[
                        valid_mask
                    ]
                    offset += count
        return temp

    def _export_vtu(
        self,
        topo: MeshTopology,
        temperatures: np.ndarray,
        filename: str,
        orig_mesh: meshio.Mesh,
    ):
        mapped = np.empty(topo.n_cells)
        hex_blocks, temp_chunks = [], []
        offset = 0
        mapped[topo.sorted_indices] = temperatures
        for block in orig_mesh.cells:
            if block.type == "hexahedron":
                count = len(block.data)
                hex_blocks.append(block)
                temp_chunks.append(mapped[offset : offset + count])
                offset += count
        meshio.Mesh(
            orig_mesh.points, hex_blocks, cell_data={"Temperature_K": temp_chunks}
        ).write(os.path.join(self.base_dir, filename))
