"""全局配置和常量定义。"""

from dataclasses import dataclass

# 全局容差配置
@dataclass(slots=True)
class Tolerances:
    abs_tol: float = 1e-6
    rel_tol: float = 1e-6
    geom_tol: float = 1e-4  # 几何比较容差 (基于配置文件中的原始长度单位)

TOL = Tolerances()