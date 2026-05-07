import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from metahotspot.metahotspot_types import SystemMatrices


class ThermalSolver:
    def __init__(self, matrices: SystemMatrices, config: dict):
        self.mat, self.config = matrices, config

    def solve_steady(self, mean_powers: np.ndarray) -> np.ndarray:
        temp = splinalg.spsolve(
            -self.mat.A_total, self.mat.b_total + (self.mat.power_matrix @ mean_powers)
        )
        print(f"[RESULT] T_min={np.min(temp):.2f} K, T_max={np.max(temp):.2f} K")
        return temp

    def solve_transient(
        self,
        dt: float,
        ptrace: list[dict],
        init_temp: np.ndarray,
        vols: np.ndarray,
        cp: np.ndarray,
    ) -> np.ndarray:
        c_mat, temp = sp.diags(cp * vols) / dt, init_temp.copy()
        solve_step = splinalg.factorized((c_mat - self.mat.A_total).tocsc())
        for i, step_power in enumerate(ptrace):
            temp = solve_step(
                (c_mat @ temp)
                + self.mat.b_total
                + (
                    self.mat.power_matrix
                    @ np.array([step_power.get(n, 0.0) for n in self.mat.unit_names])
                )
            )
            if i % 10 == 0 or i == len(ptrace) - 1:
                print(
                    f"[STEP {i:4d}] T_min={np.min(temp):.2f} K, T_max={np.max(temp):.2f} K"
                )
        return temp
