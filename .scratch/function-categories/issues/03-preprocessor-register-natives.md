---
Status: ready-for-agent
---

# 03: preprocessor 注册 native 函数

## 范围

- `src/preprocessor/preprocessor.cpp`
    - 在 `model->functions = ioStructure.functions` 之后，对每个 Function：
    1. 构造 `FieldEvaluator`（按 PRD 第 2 节给出的 5 个闭包模板）
    2. 调 `mhs::expr::register_native(name, evaluator)` 注册
    - 顺序：先注册再编译（否则编译时碰到 `name(x)` 报未注册）
- 新增 helper 文件 `src/expr/function_helpers.hpp/cpp`（或放在 preprocessor.cpp 内部）：
    - `FieldEvaluator make_expression_evaluator(const std::string& inner_expr)`
    - `FieldEvaluator make_double_exp_evaluator(double A, double alpha, double beta)`
    - `FieldEvaluator make_gauss_evaluator(double A, double tau, double x0)`
    - `FieldEvaluator make_sine_evaluator(double A, double omega, double phi)`
    - `FieldEvaluator make_piecewise_evaluator(const std::vector<PieceWiseFunction::Point>& pts)`
    - 放置策略：放 `src/expr/`，因为它们是 expr 模块的纯 helper

## 约束

- 所有 native 闭包从 `ctx.t` 取自变量（**不区分 `T` / `t`**——preprocessor 在 04 会做字面替换）
- `ExpressionFunction` 内部：内层 `parse` 自变量名仍是 `x`，闭包内构造
  `FieldContext{0,0,0, ctx.t, ctx.t}` 然后调内层 evaluator（即把 `ctx.t` 喂给内层 `x` 变量）
- 重复注册同名 → `register_native` 当前是覆盖（grep `register_native` 自检）；
  本次维持覆盖语义（重复定义 panic 由 IO 层负责——已在 02 处理）

## 验收

- 5 个 case3 闭包的数学正确性（已知参数 → 已知值；用单元测试）
- 注册后 expr 引擎能识别函数名（`parse("test_gaussian(x)")` 不报未注册）

## 不做

- 字面替换（04）
- IO 解析（02）
