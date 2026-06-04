---
Status: ready-for-agent
---

# 05: 测试与 case 回归

## 范围

### 单元测试

- 5 类函数 evaluator 数学正确性（`tests/test_function_helpers.cpp`）：
    - `make_expression_evaluator("2*x+1")` 在 `ctx.t=3` → 7
    - `make_gauss_evaluator(1, 1, 0)` 在 `ctx.t=0` → 1，`ctx.t=1` → exp(-1)
    - `make_sine_evaluator(1, 1, 0)` 在 `ctx.t=π/2` → 1
    - `make_double_exp_evaluator(1, 0.5, 0.1)` 在 `ctx.t=0` → 0
    - `make_piecewise_evaluator({(0,-1),(1,2),(5,3)})` 在 `x=-1` → -1，`x=3` → 2.5（线性），`x=10` → 3
- 字面替换正确性：
    - `test_gaussian(x)` → `test_gaussian(t)` 正确（体热源中）
    - `test_gaussian(x)` → `test_gaussian(T)` 正确（材料/BC中）
    - 孤立 x 判定：`2*x + x_next` → 第一个 `x` 替换、第二个 `x` 不替换（后跟 `_`）
    - `xx + axb` 中所有 `x` 都不替换
    - 函数名 `test_gaussian` 保持原样（不被改成 `test_gaussian_t` 之类）
    - 未注册 name → panic
- IO 解析：5 类函数都能解析；未知 type → panic

### 集成 case

- 现有 `cases/simple_steady_tests/case3.xml`：5 类函数全部被解析；结果与 reference 一致
- 现有 `cases/simple_transient_tests/case1.xml`：test_gaussian 在 TiReyuan 中以 t 代入，结果与 reference 一致
- 新增 case（如需要）：`<DaoreXishu>test_sine(T)</DaoreXishu>` 验证材料中 T 代入
- 新增 case（如需要）：`<ConvectionCoefficient>test_gaussian(T)</ConvectionCoefficient>` 验证 BC 中 T 代入

### 回归

- 现有 5 个 steady case（无函数引用）→ 数值差异 < 1e-9

## 验收

- `python run_tests.py` 全部通过
- 现有 case3.xml / case1.xml 跑通，结果在容差内
- 失败引用（未注册）→ panic，非零退出

## 不做

- 实现代码（01-04）
- 文档 / ADR（06）
