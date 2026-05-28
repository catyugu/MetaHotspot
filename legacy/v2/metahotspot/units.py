"""单位换算模块。

集中管理所有量纲的单位换算至国际标准单位 (SI)。
"""

from typing import Dict, Union
import numpy as np


class UnitConverter:
    """集中处理所有量纲的单位换算至国际标准单位 (SI)。"""

    _LENGTH_FACTORS: Dict[str, float] = {
        "M": 1.0,
        "Mm": 1e-3,   # 毫米 (上游软件拼写)
        "Um": 1e-6,   # 微米
        "Nm": 1e-9,   # 纳米
        "Inch": 0.0254, # 英寸
        "Mil": 2.54e-5, # 密耳
    }

    def __init__(self, base_length_unit: str) -> None:
        self.base_unit = base_length_unit
        self.L = self._LENGTH_FACTORS.get(base_length_unit, 1e-3)
        self.L2 = self.L ** 2
        self.L3 = self.L ** 3

    def to_m(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """长度转换: e.g., mm -> m"""
        return value * self.L
        
    def from_m(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """长度逆转换: e.g., m -> mm"""
        return value / self.L

    def to_m2(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """面积转换: e.g., mm² -> m²"""
        return value * self.L2

    def to_m3(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """体积转换: e.g., mm³ -> m³"""
        return value * self.L3