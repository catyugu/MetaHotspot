import os
import re
import meshio
import numpy as np
import time

from metahotspot.logging_config import get_logger
from metahotspot.assembler import FVMAssembler
from metahotspot.thermal_solver import ThermalSolver
from metahotspot.mesher import Mesher
from metahotspot.fluid_preprocessor import FluidPreprocessor
from metahotspot.metahotspot_types import (
    MeshTopology,
    PhysicalFields,
    BoundaryCondition,
)
from metahotspot.model25d import parse_computational_model
from metahotspot.numba_warmup import warmup_numba_kernels

_logger = get_logger(__name__)


class MetaHotspotSolver:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.base_dir = os.path.dirname(config_path)

        (
            self.solver_config,
            self.layer_regions,
        ) = parse_computational_model(config_path)

    def run(self):
        warmup_start = time.perf_counter()
        warmup_numba_kernels()
        warmup_end = time.perf_counter()
        _logger.info(
            f"Numba kernels warmup completed in {warmup_end - warmup_start:.2f} seconds"
        )

        start = time.perf_counter()
        mesher = Mesher(self.layer_regions)
        topo, fields, points, hex_cells = mesher.generate()
        mesh_gen_finished = time.perf_counter()
        _logger.info(
            f"Mesh generation & preprocessing completed in {mesh_gen_finished - start:.2f} seconds"
        )

        resolved_bcs = self._resolve_boundary_conditions(topo, fields)

        FluidPreprocessor(resolved_bcs).solve_flow(topo, fields)
        pressure_solve_finished = time.perf_counter()
        _logger.info(
            f"Fluid flow solving completed in {pressure_solve_finished - mesh_gen_finished:.2f} seconds"
        )

        matrices = FVMAssembler(
            topo, fields, resolved_bcs, self.layer_regions
        ).assemble()
        assembly_finished = time.perf_counter()
        _logger.info(
            f"System matrix assembly completed in {assembly_finished - pressure_solve_finished:.2f} seconds"
        )

        solver = ThermalSolver(matrices)
        ptrace_matrix = self._load_ptrace_matrix(matrices.unit_names)

        if self.solver_config.simulation_type == "steady":
            mean_powers = (
                np.mean(ptrace_matrix, axis=0)
                if ptrace_matrix.shape[0] > 0
                else np.zeros(len(matrices.unit_names))
            )
            temperatures = solver.solve_steady(mean_powers)
            out_filename = "result.vtu"
        else:
            temperatures = solver.solve_transient(
                self.solver_config.timestep,
                ptrace_matrix,
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
        self._export_vtu(temperatures, out_filename, points, hex_cells)

    def _resolve_boundary_conditions(
        self, topo: MeshTopology, fields: PhysicalFields
    ) -> list[BoundaryCondition]:
        resolved_bcs = []
        for cfg in self.solver_config.boundary_conditions:
            if cfg.face not in topo.boundary_faces:
                resolved_bcs.append(
                    BoundaryCondition(
                        type=cfg.type,
                        c_ids=np.array([], dtype=int),
                        areas=np.array([], dtype=float),
                        parameters=cfg.parameters,
                    )
                )
                continue

            c_ids, _, areas = topo.boundary_faces[cfg.face]

            if not cfg.target:
                resolved_bcs.append(
                    BoundaryCondition(
                        type=cfg.type,
                        c_ids=c_ids,
                        areas=areas,
                        parameters=cfg.parameters,
                    )
                )
                continue

            pattern = re.compile(cfg.target)
            layer_names = [
                fields.layer_name_map[fields.layer_ids[cid]] for cid in c_ids
            ]
            unit_names = [fields.unit_name_map[fields.unit_ids[cid]] for cid in c_ids]

            mask = np.array(
                [
                    bool(pattern.match(l_name)) or bool(pattern.match(u_name))
                    for l_name, u_name in zip(layer_names, unit_names)
                ]
            )

            resolved_bcs.append(
                BoundaryCondition(
                    type=cfg.type,
                    c_ids=c_ids[mask],
                    areas=areas[mask],
                    parameters=cfg.parameters,
                )
            )

        return resolved_bcs

    def _load_ptrace_matrix(self, unit_names: list[str]) -> np.ndarray:
        if not self.solver_config.ptrace_file_path:
            return np.zeros((0, len(unit_names)), dtype=np.float64)

        path = os.path.join(self.base_dir, self.solver_config.ptrace_file_path)
        if not os.path.exists(path):
            return np.zeros((0, len(unit_names)), dtype=np.float64)

        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return np.zeros((0, len(unit_names)), dtype=np.float64)

        headers = lines[0].split()
        name_to_idx = {name: i for i, name in enumerate(headers)}

        col_indices = [name_to_idx.get(name, -1) for name in unit_names]
        num_steps = len(lines) - 1
        power_matrix = np.zeros((num_steps, len(unit_names)), dtype=np.float64)

        for step_idx, line in enumerate(lines[1:]):
            vals = line.split()
            for u_idx, c_idx in enumerate(col_indices):
                if c_idx != -1 and c_idx < len(vals):
                    power_matrix[step_idx, u_idx] = float(vals[c_idx])

        return power_matrix

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
                    temp[offset : offset + count] = block_temps
                    offset += count
        return temp

    def _export_vtu(
        self,
        temperatures: np.ndarray,
        filename: str,
        points: np.ndarray,
        hex_cells: np.ndarray,
    ):
        meshio.Mesh(
            points,
            [("hexahedron", hex_cells)],
            cell_data={"Temperature_K": [temperatures]},
        ).write(os.path.join(self.base_dir, filename))
