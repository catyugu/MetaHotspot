# 切片 7 — 输出时刻线性插值回退

> **Status**: needs-triage
> **依赖**: 切片 6
> **阻塞**: 切片 8（真实 case 需要该回退才能跑通非整除 output_dt）

## 目标

当控制器把 dt clamp 到 `min_dt` 仍不足以命中 t_out 时，回退到线性插值。

## 修改

### `src/scheduler/scheduler.cpp::run()`

- 跟踪 `t_last` 与 `T_at_t_last`（上一步的 t 与 T）
- 当 `t_next_out` 在 `(t_last, t_last + dt_internal]` 区间内时：
    - 计算插值 `T(t_out) = T_last + (T_new - T_last) · (t_out - t_last) / dt_internal`
    - 在 `t_out` 写探针与快照

## 决策表

| 情况                                       | t_out 是否在自然落点 | 处理                |
| ------------------------------------------ | -------------------- | ------------------- |
| output_dt = 内部 dt 整数倍                 | 是                   | 直接用 `T_at_t_out` |
| 内部 dt clamp 后恰落 t_out                 | 是                   | 直接用 `T_at_t_out` |
| output_dt ≠ 内部 dt 倍数；clamp 不足以命中 | 否                   | 线性插值            |
| 控制器拒绝该步                             | —                    | 不写 t_out          |

## 测试

### `tests/test_adaptive_bdf_scheme.cpp`（补充）

- `LinearInterpFallback`：构造 `min_dt > output_dt` 强制线性插值路径
- `ProbeRecordedAtOutputTimesOnly`：构造 3 个 output_time，验证探针 times 长度 == 3
- `OutputTimesAreExact`：验证探针 times 等于 `[0, output_dt, 2·output_dt, …, < duration]`

## 验证

```bash
cmake --build build --parallel
python run_tests.py
```
