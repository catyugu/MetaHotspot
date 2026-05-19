import numpy as np
from metahotspot.assembler_kernels import (
    find_adjacent_pairs_kernel,
    overlap_area_kernel,
)


def warmup_numba_kernels():
    """使用 2 个单元的微型 Dummy 数据触发 JIT 缓存加载"""
    # 构造极简的 boxes 数据 (N, 6)
    dummy_boxes = np.array(
        [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 2.0, 1.0, 1.0]],
        dtype=np.float64,
    )

    # 触发 find_adjacent_pairs_kernel 编译/加载
    find_adjacent_pairs_kernel(dummy_boxes)

    # 触发 overlap_area_kernel
    overlap_area_kernel(dummy_boxes[0], dummy_boxes[1], 0)
