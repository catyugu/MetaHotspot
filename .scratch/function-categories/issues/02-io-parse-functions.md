---
Status: ready-for-agent
---

# 02: io.cpp 解析 \<Functions\> 块

## 范围

- `src/io/io.cpp` 在 `Materials` 解析（当前约 212-235 行）后增加 `<Functions>` 块解析
- 父节点：`<Functions xmlns:a="...">`（key 列表）
- 子节点：`<a:KeyValueOfstringFunctionAdzryM2O>` 包含 `<a:Key>` / `<a:Value i:type="b:...">`
- 5 个子类型（按 `i:type` 区分）：
    - `b:ExpressionFunction`：
        - `<b:Expression>`（字符串）
        - `<b:DrawMinX>`, `<b:DrawMaxX>`（double，可缺失 → 0/100）
    - `b:DoubleExponentialFunction`：
        - `<b:A>`, `<b:Alpha>`, `<b:Beta>`（double）
        - `<b:DrawMinX>`, `<b:DrawMaxX>`
    - `b:GaussFunction`：
        - `<b:A>`, `<b:Tau>`, `<b:X0>`（double）
        - `<b:DrawMinX>`, `<b:DrawMaxX>`
    - `b:SineFunction`：
        - `<b:A>`, `<b:Omega>`, `<b:Phi>`（double）
        - `<b:DrawMinX>`, `<b:DrawMaxX>`
    - `b:PieceWiseFunction`：
        - `<b:Points>` 内的多个 `<b:PieceWiseFunction.Point>`（每个含 `<b:X>`, `<b:Y>`）
        - `<b:DrawMinX>`, `<b:DrawMaxX>`
- 全部解析到 `IOStructure.functions[name] = Function{...}`
- 缺失字段 → 默认 0.0（与 `BiRerong i:nil="true"` 一致）
- 未知 `i:type` → panic（明确报错，不要静默）

## 约束

- 不改 `read_xml` 其它部分
- 不改 write 路径

## 验收

- case3.xml（5 类函数）解析出 `functions.size() == 5`，每类一个
- case1.xml（仅 Gauss）解析出 `functions.size() == 1`
- 没有 `<Functions>` 块的 case → `functions` 为空 map
- 未知 type → panic，错误信息含 type 名

## 不做

- preprocessor 注册（03）
- 字面替换（04）
