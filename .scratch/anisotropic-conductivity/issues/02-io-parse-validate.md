---
Status: ready-for-agent
---

# 02: io::read_xml 解析并校验 DaoreXishu

## 范围

- `src/io/io.cpp`（约 212-235 行的 `Materials` 解析）
    - `<DaoreXishu>` 文本按 `,` 分割，每段 `trim` 空白
    - 段数校验：
        - 1 → `kx = ky = kz = segments[0]`
        - 3 → `kx = segments[0], ky = segments[1], kz = segments[2]`
        - 0 / 2 / ≥4 → `panic("DaoreXishu must have 1 or 3 comma-separated expressions, got N: '<原文>'")`
    - 段为空（trim 后空串）→ 同样 panic
- 注意：当前 XML 元素是单行文本节点，确保 `get_text` 取到整段字符串再做分割

## 失败语义

- 用 `mhs::logger::panic` —— 现有 io::read_xml 错误即退出码非零、不会继续到装配
- 错误信息要带原文（截断 ≤ 200 字符防日志爆）

## 验收

- 三段（k1, k2, k3）解析正确
- 单段（k1）解析为 kx=ky=kz=k1
- 两段 / 四段 / 空段：panic 触发，CI 单测覆盖
- 单表达式现状 case（case1/2/3 等）继续能解析

## 不做

- `preprocessor` 编译（属于 03）
- 装配 / 后处理（属于 04）
