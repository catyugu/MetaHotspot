---
Status: ready-for-human
---

# 06: 文档与 ADR

## 范围

- `docs/adr/0007-transient-probe-output.md`
    - Context：现有瞬态循环无输出；探针坐标 XML 已有
    - Decision：StepCallback 回调 + sample_point 复用 LSQ 插值
    - Consequences：每步插值开销、trace 数据结构、write_xml 扩展
- `docs/design/io-model.md`
    - 更新：`ObservationPoint3D` / `ProbeTrace` 说明
- `docs/design/data-flow.md`
    - 更新瞬态流程图：加入 on_step_done 回调
- `CONTEXT.md`
    - 术语表加一行：ObservePoint3D = 观察点（3D 探针坐标）
    - 术语表加一行：ProbeTrace = 观察点温度时间序列

## 验收

- ADR 有完整 Context / Decision / Consequences 三段
- grep `ObservePoints3D` 在 docs/ 中有说明

## 不做

- 代码 / 测试（01-05）
