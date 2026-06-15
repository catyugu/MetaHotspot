# 09: 文档（ADR-0006 / design / CONTEXT.md）

Status: needs-triage · Type: docs · Depends on: 06, 07, 08

## Context

新增功能需要沉淀三处文档：

1. `docs/adr/0006-fluid-solid-interface-override.md` — 流-固边界覆盖策略的 ADR
2. `docs/design/fluid-solver.md` — 流体解算与装配器集成的设计文档
3. `CONTEXT.md` 术语表 — FluidMaterial / PressureBC / FluidSolidInterface / FluidFields 等

## Goal

按本仓库 ADR / 设计文档 / 术语表的现有格式输出三份文档。

### ADR-0006：流-固边界覆盖策略

参照 `docs/adr/0005-cell-level-bc.md` 格式：

```markdown
# ADR-0006: Fluid-Solid Interface Override

## Status
Accepted.

## Context
流体解算引入后，流-固交界面（c0.is_fluid XOR c1.is_fluid）需要被
Nusselt 关联式自动给出的 h 系数覆盖，而不是用户可能配置的 cauchy /
dirichlet / neumann BC。物理理由是：在 Poiseuille 假设下，固液传热
是流体解的产物，不应被静态边界条件覆盖。

## Decision
- preprocessor 检测流-固面，写入 FluidFields.fs_faces
- 装配器在面循环开头查 fs_faces；命中即只应用 h 项，continue 跳过
  扩散 / cauchy / dirichlet 等一切其它贡献
- 用户侧在 face_key 配置流-固面其它 BC 不会被警告（不与 GUI 冲突）；
  静默忽略

## Rationale
- 与 Poiseuille 假设自洽
- 不污染用户 BC 路径
- GUI 短期不感知

## Data flow
（沿用 ADR-0005 风格的图）

## Notes
- fs_faces 在 preprocessor 阶段算好；装配阶段只查表
- 装配阶段 fs_faces 索引用 std::unordered_map<int, size_t> 缓存
```

### docs/design/fluid-solver.md

参照 `docs/design/internal-model.md` 风格，章节：

1. 数据流图（主 XML + sidecar → InternalModel → Scheduler）
2. FluidFields 完整定义
3. Sidecar XML schema（含 BNF / 示例）
4. Pressure 方程装配（CSR 三元组细节）
5. Poiseuille hydroC 公式 + 矩形 Nu 关联式
6. 体积对流项离散（迎风格式细节）
7. 流-固面 h 项细节
8. 与热装配的耦合点（face 循环位置图）
9. 错误处理矩阵

### CONTEXT.md 术语表增量

在"术语表"章节按表格新增：

| 概念                | 中文         | 说明                                                              |
| ------------------- | ------------ | ----------------------------------------------------------------- |
| FluidMaterial       | 流体材料     | Material 携带 is_fluid=true + dynamic_viscosity 表达式            |
| PressureBC          | 压力边界     | sidecar `<PressureBoundary>`；bc_params.pressure_p                |
| FluidSolidInterface | 流-固交界面  | c0.is_fluid XOR c1.is_fluid；自动检测；h 由 Nu 公式给出           |
| FluidFields         | 流体场       | InternalModel.fluid；pressure / hydroC / face_velocity / fs_faces |
| FluidPreprocessor   | 流体预处理器 | solve_flow(model)；Preprocessor::load 末尾调用                    |
| hydroC              | 水力传导     | m^3·s/kg；Poiseuille 解析解每轴标量                               |
| Sidecar XML         | 附加声明     | `<name>_additional.xml`；与主 XML 同目录                          |
| mu_ref              | 参考粘度     | T_ref（默认 initial_temperature）下 μ 表达式求值结果              |

## Scope

- 三份文档，**纯文本**；不写代码、不改代码
- 引用代码路径时用 `src/...` 真实路径
- ADR-0006 编号紧接现有 0005

## Acceptance

1. `docs/adr/0006-fluid-solid-interface-override.md` 存在，格式与 0005 对齐
2. `docs/design/fluid-solver.md` 存在，覆盖上面 9 个章节
3. `CONTEXT.md` 术语表含上述 8 条新术语
4. `git grep` 验证：所有术语在代码 / 文档里使用一致（无别名漂移）

## Notes

- ADR 不必"惊喜"才写；覆盖策略本身就是反直觉的（用户写 cauchy 被忽略），值得 ADR
- 不在本 issue 改测试 / 改代码
- 术语命名严格用表格中的英文 / 中文，避免在文档 / 代码 / 测试间漂移
