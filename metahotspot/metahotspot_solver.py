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
from metahotspot.model25d import load_config, load_stackup
from metahotspot.numba_warmup import warmup_numba_kernels

_logger = get_logger(__name__)


class MetaHotspotSolver:
    def __init__(self, config_path: str):
        self.config_path, self.base_dir = config_path, os.path.dirname(config_path)
        self.config, self.stackup = load_config(config_path), load_stackup(
            load_config(config_path), os.path.dirname(config_path)
        )
        self.mesh_path = os.path.join(self.base_dir, self.config["mesh_file_path"])

    def run(self):
        warmup_start = time.perf_counter()
        warmup_numba_kernels()
        warmup_end = time.perf_counter()
        _logger.info(
            f"Numba kernels warmup completed in {warmup_end - warmup_start:.2f} seconds"
        )
        start = time.perf_counter()
        topo, fields = MeshPreprocessor(self.config, self.stackup).process(
            self.mesh_path
        )
        mesh_finished = time.perf_counter()
        _logger.info(
            f"Mesh preprocessing completed in {mesh_finished - start:.2f} seconds"
        )
        FluidPreprocessor(self.config).solve_flow(topo, fields)
        pressure_solve_finished = time.perf_counter()
        _logger.info(
            f"Fluid flow solving completed in {pressure_solve_finished - mesh_finished:.2f} seconds"
        )
        matrices = FVMAssembler(topo, fields, self.config, self.stackup).assemble()
        assembly_finished = time.perf_counter()
        _logger.info(
            f"System matrix assembly completed in {assembly_finished - pressure_solve_finished:.2f} seconds"
        )
        solver, ptrace = ThermalSolver(matrices, self.config), self._load_ptrace()
        if self.config["simulation_type"] == "steady":
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

        else:
            temperatures = solver.solve_transient(
                self.config["timestep"],
                ptrace,
                self._get_init_temp(topo),
                topo.volumes,
                fields.cp,
            )
        end = time.perf_counter()
        _logger.info(
            f"Thermal solving completed in {end - assembly_finished:.2f} seconds"
        )
        _logger.info(f"Simulation completed in {end - start:.2f} seconds")
        _logger.info("Exporting results...")
        self._export_vtu(topo, temperatures, "transient_result.vtu")

    def _load_ptrace(self) -> list[dict]:
        path = os.path.join(self.base_dir, self.config.get("ptrace_file_path", ""))
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            headers = f.readline().split()
            return [dict(zip(headers, map(float, l.split()))) for l in f if l.strip()]

    def _get_init_temp(self, topo: MeshTopology) -> np.ndarray:
        temp = np.full(topo.n_cells, float(self.config["init_temperature"]))
        init_file = self.config.get("init_temperature_file_path")
        if init_file and os.path.exists(os.path.join(self.base_dir, init_file)):
            init_mesh, offset = meshio.read(os.path.join(self.base_dir, init_file)), 0
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

    def _export_vtu(self, topo: MeshTopology, temperatures: np.ndarray, filename: str):
        mapped, orig_mesh, hex_blocks, temp_chunks, offset = (
            np.empty(topo.n_cells),
            meshio.read(self.mesh_path),
            [],
            [],
            0,
        )
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
