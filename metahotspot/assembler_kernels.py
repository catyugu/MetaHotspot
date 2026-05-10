"""
Numba compute kernels for FVM assembly.
计算与状态分离 - OOP 层负责数据解包，Kernel 负责计算。
"""

from numba import njit
import numpy as np

FLOAT_DTYPE = np.float64
INT_DTYPE = np.int32


def _jit_kernel(func):
    return njit(cache=True, fastmath=True, nogil=True)(func)


@_jit_kernel
def overlap_area_kernel(box_a, box_b, axis):
    if axis == 0:
        d1 = min(box_a[4], box_b[4]) - max(box_a[1], box_b[1])
        d2 = min(box_a[5], box_b[5]) - max(box_a[2], box_b[2])
    elif axis == 1:
        d1 = min(box_a[3], box_b[3]) - max(box_a[0], box_b[0])
        d2 = min(box_a[5], box_b[5]) - max(box_a[2], box_b[2])
    else:
        d1 = min(box_a[3], box_b[3]) - max(box_a[0], box_b[0])
        d2 = min(box_a[4], box_b[4]) - max(box_a[1], box_b[1])
    return d1 * d2 if d1 > 0.0 and d2 > 0.0 else 0.0


@_jit_kernel
def find_adjacent_pairs_kernel(boxes):
    n_cells = boxes.shape[0]
    max_pairs = n_cells * 6

    c_a_arr = np.empty(max_pairs, dtype=INT_DTYPE)
    c_b_arr = np.empty(max_pairs, dtype=INT_DTYPE)
    axis_arr = np.empty(max_pairs, dtype=INT_DTYPE)
    area_arr = np.empty(max_pairs, dtype=FLOAT_DTYPE)

    ptr = 0
    tol = 1e-12  # 工业常用容差
    sorted_ids = np.argsort(boxes[:, 0])

    for i in range(len(sorted_ids)):
        c_a = sorted_ids[i]
        for j in range(i + 1, len(sorted_ids)):
            c_b = sorted_ids[j]
            if boxes[c_b, 0] > boxes[c_a, 3] + tol:
                break

            b_a = boxes[c_a]
            b_b = boxes[c_b]

            if (
                max(b_a[1], b_b[1]) > min(b_a[4], b_b[4]) + tol
                or max(b_a[2], b_b[2]) > min(b_a[5], b_b[5]) + tol
            ):
                continue

            for axis in range(3):
                if (
                    abs(b_a[axis + 3] - b_b[axis]) < tol
                    or abs(b_a[axis] - b_b[axis + 3]) < tol
                ):
                    area = overlap_area_kernel(b_a, b_b, axis)
                    if area > tol:
                        c_a_arr[ptr] = c_a
                        c_b_arr[ptr] = c_b
                        axis_arr[ptr] = axis
                        area_arr[ptr] = area
                        ptr += 1

    return c_a_arr[:ptr], c_b_arr[:ptr], axis_arr[:ptr], area_arr[:ptr], ptr


@_jit_kernel
def compute_nusselt_kernel(dim_w, dim_h):
    w, h = min(dim_w, dim_h), max(dim_w, dim_h)
    AR = w / h if h > 0 else 1.0
    return 8.235 * (
        1
        - 2.0421 * AR
        + 3.0853 * AR**2
        - 2.4765 * AR**3
        + 1.0578 * AR**4
        - 0.1861 * AR**5
    )


@_jit_kernel
def build_cond_coo_kernel(
    c_a_arr, c_b_arr, axis_arr, area_arr, count, dims, k_arr, is_fluid, flow_axes
):
    rows = np.empty(count * 4, dtype=INT_DTYPE)
    cols = np.empty(count * 4, dtype=INT_DTYPE)
    data = np.empty(count * 4, dtype=FLOAT_DTYPE)
    ptr = 0

    for i in range(count):
        c_a, c_b, axis, area = c_a_arr[i], c_b_arr[i], axis_arr[i], area_arr[i]
        fluid_a, fluid_b = is_fluid[c_a], is_fluid[c_b]

        if fluid_a != fluid_b:
            f_id, s_id = (c_a, c_b) if fluid_a else (c_b, c_a)
            f_ax = flow_axes[f_id]
            ax_w, ax_h = (f_ax + 1) % 3, (f_ax + 2) % 3
            Nu = compute_nusselt_kernel(dims[f_id, ax_w], dims[f_id, ax_h])
            d_h = (
                2
                * dims[f_id, ax_w]
                * dims[f_id, ax_h]
                / (dims[f_id, ax_w] + dims[f_id, ax_h])
                if (dims[f_id, ax_w] + dims[f_id, ax_h]) > 0
                else 1.0
            )
            h_f = (Nu * k_arr[f_id]) / d_h if d_h > 0 else 1e-6
            r = dims[s_id, axis] / (2.0 * k_arr[s_id] * area) + 1.0 / (h_f * area)
        else:
            r = (dims[c_a, axis] / (2.0 * k_arr[c_a] * area)) + (
                dims[c_b, axis] / (2.0 * k_arr[c_b] * area)
            )

        g = 1.0 / r if r > 0 else 0.0

        rows[ptr : ptr + 4] = (c_a, c_b, c_a, c_b)
        cols[ptr : ptr + 4] = (c_a, c_b, c_b, c_a)
        data[ptr : ptr + 4] = (-g, -g, g, g)
        ptr += 4

    return rows[:ptr], cols[:ptr], data[:ptr]


@_jit_kernel
def build_adv_coo_kernel(
    c_a_arr, c_b_arr, axis_arr, count, is_fluid, pressure, density, hydroC, cp
):
    rows = np.empty(count * 2, dtype=INT_DTYPE)
    cols = np.empty(count * 2, dtype=INT_DTYPE)
    data = np.empty(count * 2, dtype=FLOAT_DTYPE)
    net_outflux = np.zeros(pressure.shape[0], dtype=FLOAT_DTYPE)
    ptr = 0
    tol = 1e-12

    for i in range(count):
        c_a, c_b, axis = c_a_arr[i], c_b_arr[i], axis_arr[i]
        if not (is_fluid[c_a] and is_fluid[c_b]):
            continue

        hc_a, hc_b = hydroC[c_a, axis], hydroC[c_b, axis]
        sum_hc = hc_a + hc_b
        C_eff = (2.0 * hc_a * hc_b / sum_hc) if sum_hc > 0 else 0.0

        mass_flux = (
            (pressure[c_a] - pressure[c_b])
            * C_eff
            * (density[c_a] + density[c_b])
            * 0.5
        )
        net_outflux[c_a] += mass_flux
        net_outflux[c_b] -= mass_flux

        if abs(mass_flux) > tol:
            up, dn = (c_a, c_b) if mass_flux > 0 else (c_b, c_a)
            adv = abs(mass_flux) * cp[up]
            rows[ptr : ptr + 2] = (up, dn)
            cols[ptr : ptr + 2] = (up, up)
            data[ptr : ptr + 2] = (-adv, adv)
            ptr += 2

    return rows[:ptr], cols[:ptr], data[:ptr], net_outflux
