# 切片 5 — IO `TimeSchemeSpec`

> **Status**: needs-triage
> **依赖**: 切片 0–3（至少 Scheduler 已切到 TimeScheme）
> **可与切片 4 并行**: 是（改不同文件）

## 目标

XML 配置支持 `<Transient>` 子节点 `<Scheme>`；老 XML 缺节点时默认 BDF1。

## 修改

### `src/data/io_model.hpp`

- 新增 `enum class TimeSchemeKind { Bdf1, Bdf2, AdaptiveBdf }`
- 新增 `struct TimeSchemeSpec { kind; initial_dt; min_dt; max_dt; abs_tol; rel_tol; max_order; output_dt; }`
- `IOStructure` 加 `TimeSchemeSpec time_scheme;` 字段

### `src/io/io.cpp`

- 解析 `<Transient>` 下 `<Scheme>`、`<InitialDt>`、`<MinDt>`、`<MaxDt>`、`<AbsTol>`、`<RelTol>`、`<MaxOrder>`、`<OutputDt>`
- 缺节点时：`kind=Bdf1`、`initial_dt=transient_time_step`、`output_dt=transient_duration`

### `src/data/internal_model.hpp`

- 加 `TimeSchemeSpec time_scheme;`
- 保留 `transient_time_step`、`transient_duration`（**仅供 IO 输出**读取用，scheduler 不依赖）
- 新注释：deprecated on algorithm side, kept for IO compat only

### `src/preprocessor/preprocessor.cpp`

- 把 `IOStructure::time_scheme` 翻译为 `InternalModel::time_scheme`

### `src/scheduler/scheduler.cpp`

- `run()` 初始化时按 `model_->time_scheme.kind` 工厂创建 TimeScheme

## 测试

### `tests/test_io.cpp`

- `SchemeDefaultsToBdf1`：老 XML 无 `<Scheme>` → `time_scheme.kind == Bdf1`
- `ParsesSchemeNode`：新 XML `<Scheme>AdaptiveBdf</Scheme>` 解析正确
- `ParsesAllKnobs`：所有新节点解析正确
- `OutputDtDefaultsToDuration`：`output_dt` 缺省 = `transient_duration`

## 验证

```bash
python run_tests.py
python run_cases.py  # 确认老 XML 仍跑通
```
