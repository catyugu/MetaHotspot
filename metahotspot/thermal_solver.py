import numpy as np
import scipy.sparse as sp
import time

from metahotspot.logging_config import get_logger
from metahotspot.metahotspot_types import SystemMatrices

import scipy.sparse.linalg as splinalg

_logger = get_logger(__name__)


class ThermalSolver:
    def __init__(self, matrices: SystemMatrices):
        # 移除弱类型字典 config 的传入，强依赖于装配阶段的 SystemMatrices
        self.mat = matrices

    def solve_steady(self, mean_powers: np.ndarray) -> np.ndarray:
        rhs = self.mat.b_total + (self.mat.power_matrix @ mean_powers)
        A = -self.mat.A_total.tocsr()

        t0 = time.perf_counter()
        temp = splinalg.spsolve(A, rhs)

        _logger.info(
            f"Steady solve took {time.perf_counter() - t0:.3f}s. T_min={np.min(temp):.2f} K, T_max={np.max(temp):.2f} K"
        )
        return temp

    def solve_transient(
        self,
        dt: float,
        ptrace: list[dict],
        init_temp: np.ndarray,
        vols: np.ndarray,
        cp: np.ndarray,
    ) -> np.ndarray:
        c_mat = sp.diags(cp * vols) / dt
        A_step = c_mat - self.mat.A_total
        temp = init_temp.copy()
        solve_func = splinalg.factorized(A_step.tocsc())

        for i, step_power in enumerate(ptrace):
            power_vec = np.array([step_power.get(n, 0.0) for n in self.mat.unit_names])
            rhs = (
                (c_mat @ temp) + self.mat.b_total + (self.mat.power_matrix @ power_vec)
            )

            temp = solve_func(rhs)

            if i % 10 == 0 or i == len(ptrace) - 1:
                _logger.info(
                    f"Step {i:4d}: T_min={np.min(temp):.2f} K, T_max={np.max(temp):.2f} K"
                )
        return temp
