---
Status: ready-for-agent
---

# 03: preprocessor 编译 kx / ky / kz

## 范围

- `src/preprocessor/preprocessor.cpp`（约 77-83 行 material_table 编译循环）
    - 改为 `expr::parse(mat.kx) / .ky / .kz` 三个
    - 任意一个失败 → panic（不要走 `parse` 的常量回退 0.0）
        - 实际：`expr::parse` 当前会回退为 0.0 常量，要绕过：
            - 显式调用 `make_constant` 之外的入口或增加 `try_parse` 探测；
            - 或者在 parse 之前用 `strtod` 检测为数字又 OK（已有逻辑），非数字则调用一个会
        抛出 / 返回 bool 的 test-compile
    - 解决方案任选其一，但**禁止在编译失败时静默回退到 0**——这是 panic 条件

## 验收

- 三段非平凡表达式（`"400 + 0.01*T"`、`"1.3 * x"`、`"130 + 0.5*z"`）全部编译通过
- 任一字段非法表达式 → panic，错误信息含字段名（kx/ky/kz）+ 表达式原文
- 单表达式 case（改前等价的）继续编译正确

## 不做

- 装配 / 后处理（04）
- IO 解析（02）
