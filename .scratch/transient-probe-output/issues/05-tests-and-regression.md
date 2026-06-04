---
Status: ready-for-agent
---

# 05: 测试与 case 回归

## 范围

### 单元测试

- `sample_point` 单测（`tests/test_postprocessor.cpp` 或新增）：
    - 均匀温度场 → 任意内部点 ≈ 300K
    - 网格外点 → NaN
    - 线性梯度场 → 插值误差 < 1e-6
- `read_xml` 单测（`tests/test_io.cpp`）：
    - 有 `<ObservePoints3D>` → observation_points 正确解析
    - 无 `<ObservePoints3D>` → observation_points 为空
- `write_xml` 单测：
    - traces 非空 → 输出 XML 含 Result0DTransient
    - traces 空 → 不动（稳态不变）
    - 已有 PointName → 清空重写

### 回归验证

- 5 个 steady case → 数值差异 < 1e-9（无改动路径）
- 瞬态 case1.xml → 末步温度场 + 观察点轨迹与原 reference 值在 1e-3 内一致

## 验收

- `python run_tests.py` 全部通过
- 稳态回归无损
- 瞬态探针值在容差内

## 不做

- 实现代码（01-04）
- 文档 / ADR（06）
