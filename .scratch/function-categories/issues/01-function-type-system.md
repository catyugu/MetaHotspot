---
Status: ready-for-agent
---

# 01: 实现 Function 类型体系 + IOStructure 字段

## 范围

- `src/common/io_model.hpp`
    - 引入 `FunctionType` 枚举：`Expression / DoubleExponential / Gauss / Sine / PieceWise`
    - 引入 5 个 POD struct（与 docs/design/io-model.md 第 110-156 行一致）：
        - `ExpressionFunction`：`std::string expression; double draw_min_x, draw_max_x;`
        - `DoubleExponentialFunction`：`double a, alpha, beta, draw_min_x, draw_max_x;`
        - `GaussFunction`：`double a, tau, x0, draw_min_x, draw_max_x;`
        - `SineFunction`：`double a, omega, phi, draw_min_x, draw_max_x;`
        - `PieceWiseFunction`：`struct Point { double x, y; }; std::vector<Point> points; double draw_min_x, draw_max_x;`
    - 引入 `Function` 联合（union + 标识字段；或 struct with all 字段 + type 枚举）：

    ```cpp
    struct Function {
        std::string key;
        FunctionType type;
        ExpressionFunction       expression;
        DoubleExponentialFunction double_exp;
        GaussFunction            gauss;
        SineFunction             sine;
        PieceWiseFunction        piecewise;
    };
    ```

    - `IOStructure.functions` 类型从 `unordered_map<string, FieldEvaluator>` 改为 `unordered_map<string, Function>`
- `src/common/internal_model.hpp`
    - `InternalModel.functions` 同样改为 `unordered_map<string, Function>`
- 现有 `IOStructure.functions: unordered_map<string, FieldEvaluator>` **彻底删除**，
  不要保留 shim

## 约束

- POD / 纯值类型
- 默认构造有效
- 不改 `FieldContext` / `FieldEvaluator` / `CompiledExpression`

## 验收

- 编译通过（此时 io.cpp / preprocessor 还没用新字段，先用 stub）

## 不做

- io.cpp 解析（02）
- preprocessor 注册（03）
- 字面替换（04）
