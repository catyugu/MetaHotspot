---
Status: ready-for-human
---

# 06: 文档与 ADR

## 范围

- `docs/adr/0006-anisotropic-conductivity.md`
    - 决策：单表达式兼容、三表达式原生支持、错误模式
    - 后果：装配 / 后处理按 FaceDir 选分量；MaterialProps 字段拆分；测试覆盖
- `docs/design/io-model.md`
    - 更新 `Material` 定义，注释 DaoreXishu 的 1/3 表达式语法
- `CONTEXT.md`（如需要）
    - 术语表加一行：kx/ky/kz = 方向化导热系数

## 验收

- 新增 ADR 文件有完整的"Context / Decision / Consequences"三段
- 现有文档无残留旧字段名（grep `daore_xishu` 不出现在 docs/）

## 不做

- 代码 / 测试（属于 02-05）
