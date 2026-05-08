"""
Numba compute kernels for FVM assembly.
计算与状态分离 - OOP 层负责数据解包，Kernel 负责计算。
"""

from numba import njit
import numpy as np

# ==========================================
# 类型别名 - 确保类型稳定性
# ==========================================
FLOAT_DTYPE = np.float64
INT_DTYPE = np.int32


# ==========================================
# Numba 装饰器工厂 - 统一配置
# ==========================================
def _jit_kernel(func):
    """工业标准装饰器: cache + fastmath + nogil"""
    return njit(cache=True, fastmath=True, nogil=True)(func)


@_jit_kernel
def overlap_area_kernel(
    box_a: FLOAT_DTYPE,
    box_b: FLOAT_DTYPE,
    axis: INT_DTYPE,
) -> FLOAT_DTYPE:
    """
    计算两个包围盒在指定轴法向上的重叠面积。

    Parameters
    ----------
    box_a : np.ndarray, shape=(6,)
        [x_min, y_min, z_min, x_max, y_max, z_max]
    box_b : np.ndarray, shape=(6,)
        同上
    axis : int
        0=X, 1=Y, 2=Z（取垂直于该轴的两个面）

    Returns
    -------
    float
        重叠面积 [m^2]
    """
    # 三个轴对应的面索引
    # axes 格式: (排除轴的最小坐标索引, 排除轴的次小坐标索引, 排除轴的最大坐标索引, 排除轴的最大坐标索引)
    if axis == 0:
        # Y-Z plane: 排除 X 轴
        d1 = min(box_a[4], box_b[4]) - max(box_a[1], box_b[1])
        d2 = min(box_a[5], box_b[5]) - max(box_a[2], box_b[2])
    elif axis == 1:
        # X-Z plane: 排除 Y 轴
        d1 = min(box_a[3], box_b[3]) - max(box_a[0], box_b[0])
        d2 = min(box_a[5], box_b[5]) - max(box_a[2], box_b[2])
    else:
        # X-Y plane: 排除 Z 轴
        d1 = min(box_a[3], box_b[3]) - max(box_a[0], box_b[0])
        d2 = min(box_a[4], box_b[4]) - max(box_a[1], box_b[1])
    return d1 * d2 if d1 > 0.0 and d2 > 0.0 else 0.0


# ==========================================
# 数组 Kernel（邻近单元查找 - Sweep and Prune）
# ==========================================


@_jit_kernel
def find_adjacent_pairs_kernel(boxes):
    """
    Sweep-and-prune 邻近单元查找。

    预分配策略：每个单元最多 6 个相邻面
    max_pairs = n_cells * 6

    Parameters
    ----------
    boxes : np.ndarray, shape=(n_cells, 6)
        [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns
    -------
    c_a_arr : np.ndarray
    c_b_arr : np.ndarray
    axis_arr : np.ndarray
    area_arr : np.ndarray
    count : int
    """
    n_cells = boxes.shape[0]
    max_pairs = n_cells * 6

    c_a_arr = np.empty(max_pairs, dtype=np.int32)
    c_b_arr = np.empty(max_pairs, dtype=np.int32)
    axis_arr = np.empty(max_pairs, dtype=np.int32)
    area_arr = np.empty(max_pairs, dtype=np.float64)

    ptr = 0
    tol = 1e-15

    sorted_ids = np.argsort(boxes[:, 0])

    for i in range(len(sorted_ids)):
        c_a = sorted_ids[i]
        for j in range(i + 1, len(sorted_ids)):
            c_b = sorted_ids[j]
            if boxes[c_b, 0] > boxes[c_a, 3] + tol:
                break

            b_a = boxes[c_a]
            b_b = boxes[c_b]

            # BBox Y/Z 排斥校验
            if (
                max(b_a[1], b_b[1]) > min(b_a[4], b_b[4]) + tol
                or max(b_a[2], b_b[2]) > min(b_a[5], b_b[5]) + tol
            ):
                continue

            # 面接触检查
            for axis in range(3):
                if (
                    abs(b_a[axis + 3] - b_b[axis]) < tol
                    or abs(b_a[axis] - b_b[axis + 3]) < tol
                ):
                    # 计算重叠面积
                    if axis == 0:
                        d1 = min(b_a[4], b_b[4]) - max(b_a[1], b_b[1])
                        d2 = min(b_a[5], b_b[5]) - max(b_a[2], b_b[2])
                    elif axis == 1:
                        d1 = min(b_a[3], b_b[3]) - max(b_a[0], b_b[0])
                        d2 = min(b_a[5], b_b[5]) - max(b_a[2], b_b[2])
                    else:
                        d1 = min(b_a[3], b_b[3]) - max(b_a[0], b_b[0])
                        d2 = min(b_a[4], b_b[4]) - max(b_a[1], b_b[1])
                    area = d1 * d2 if d1 > 0.0 and d2 > 0.0 else 0.0
                    if area > tol:
                        c_a_arr[ptr] = c_a
                        c_b_arr[ptr] = c_b
                        axis_arr[ptr] = axis
                        area_arr[ptr] = area
                        ptr += 1

    return c_a_arr[:ptr], c_b_arr[:ptr], axis_arr[:ptr], area_arr[:ptr], ptr
