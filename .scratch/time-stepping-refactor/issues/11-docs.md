# 切片 10 — 文档更新

> **Status**: needs-triage
> **依赖**: 切片 0–9
> **阻塞合并**: 是（文档与代码脱节会误导后续维护者）

## 目标

反映新架构；标记 deprecated 字段。

## 修改

### `CONTEXT.md`

- "求解流程" 章节：更新 `Scheduler::run()` 流程图，加入 `TimeScheme` 与 `TimeStepBuffer`
- "关键设计原则" 加一条：算法与组装解耦

### `docs/design/module-interfaces.md`

- 更新 `assembler` 段：暴露 `assemble_static` + `assemble_mass`
- 新增 `time_scheme` 段：TimeScheme、Bdf1Scheme、Bdf2Scheme、AdaptiveBdfScheme

### `docs/design/data-flow.md`

- 更新"求解"段：主循环走 TimeScheme

### `docs/adr/0006-time-stepping.md`

- 状态从 "Proposed" 改为 "Accepted"

### `docs/design/internal-model.md`

- `GlobalState` 字段更新（`T_prev` → `history`）

### `src/data/internal_model.hpp::GlobalState` 注释

- 明确 `transient_time_step`、`transient_duration` 仅供 IO 输出

## 验证

文档与代码一致。
