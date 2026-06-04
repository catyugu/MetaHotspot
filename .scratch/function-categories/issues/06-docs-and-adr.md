---
Status: ready-for-human
---

# 06: 文档与 ADR

## 范围

- `docs/adr/0008-function-categories.md`
    - Context：当前 Functions 块未解析，引用 `name(x)` 在 exprtk 失败
    - Decision：5 类单变元函数 native 注册 + 字面替换 name(x)→name(t or T)
    - Consequences：preprocessor 阶段一次性代价、字面替换的 regex 脆弱性
- `docs/design/io-model.md`
    - 把第 110-156 行的 Function 类型体系从"计划"挪到"已实现"标注
- `CONTEXT.md`
    - 术语表加：Functions = 单变元函数字典（5 类）
    - 术语表加：自变量映射（体热源→t，材料/BC→T）

## 验收

- ADR 完整三段
- docs/design/io-model.md 第 110-156 行的"TODO:尚未定义"标注移除

## 不做

- 代码 / 测试（01-05）
