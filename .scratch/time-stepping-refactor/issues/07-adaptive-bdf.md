# 切片 6 — `AdaptiveBdfScheme` + 控制器

> **Status**: needs-triage
> **依赖**: 切片 0–5
> **可与切片 7 串行**: 是

## 目标

自适应阶变步长 BDF。控制器选择每步的 (dt, order)，并决策接受/拒绝。

## 新建

- `src/time_scheme/adaptive_bdf_scheme.hpp` / `.cpp`
- `src/time_scheme/step_controller.hpp` / `.cpp`（控制器可独立单元测试）

## 控制器逻辑

参考 Hairer-Norsett-Wanner 卷 II §V：

1. 用 order k 与 order k-1 两次预测的差 `e = T^{(k)} - T^{(k-1)}` 作为局部截断误差估计
2. 决策：
   - `||e||_∞ ≤ tol` → 接受；`order_new = min(k+1, max_order)`，`dt_new = safety · dt · (tol/||e||)^{1/(k+1)}`
   - `||e|| > tol` → 拒绝；`order_new = k`（保持或降 1），`dt_new = safety · dt · (tol/||e||)^{1/k}`，clamp 到 `[min_dt, max_dt]`

## δ 范围约束

默认 `0.5 ≤ δ ≤ 2.0` 软约束；超出时按公式自动衰减（不必显式拒绝）。

## 输出时刻对齐（本切片只做 clamp，回退放切片 7）

- 维护 `t_next_output = output_step · output_dt`
- 每步 `dt_internal = min(dt_internal, t_next_out - t_current)`
- 当 `t ≈ t_next_output`（容差 `1e-9·max(1, t)`）时：写探针 + `output_step++` + `t_next_output += output_dt`

## Scheduler 接入

`scheduler.cpp` 需跟踪 `t_last` 与 `T_at_t_last`（为切片 7 准备）。

## 测试

### `tests/test_adaptive_bdf_scheme.cpp`

- `ShrinksOnLargeError`：构造 T 使 k=2 预测与 k=1 预测差大 → `accept_or_reject == Reject`
- `GrowsOnSmallError`：构造 T 使误差小 → `Accept` + 下次 dt 增大
- `ClampsToMinDt`：dt 触底时 clamp 到 `min_dt`
- `ClampsToMaxDt`：dt 触顶时 clamp 到 `max_dt`
- `StepLogRecordsEverything`：step_log 字段记录每步 (t, dt, order, verdict)

### `tests/test_step_controller.cpp`

- 单元测试控制器对误差的 dt 选择对数斜率正确

## 验证

```bash
cmake --build build --parallel
python run_tests.py
```
