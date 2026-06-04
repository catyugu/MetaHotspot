---
Status: ready-for-agent
---

# 04: 字面替换 name(x) → name(t or T) + 编译

## 范围

- `src/preprocessor/preprocessor.cpp`
    - 新增 helper：`std::string substitute_function_args(const std::string& expr_str, const std::string& argname, const std::unordered_map<std::string, Function>& fns)`
        - 遍历 fns 的 name（按 name 长度倒序，避免短名误替换长名）
        - 替换 name 不动（`test_gaussian` 仍是 `test_gaussian`）
        - 关键：**自变量 `x` 的判定** — 扫描 expr_str 中每个字符 `x`，仅当
      `(i == 0 || (!isalpha(expr_str[i-1]) && expr_str[i-1] != '_')) &&
       (i+1 == n || (!isalpha(expr_str[i+1]) && expr_str[i+1] != '_'))`
      成立时才视作"孤立自变量 x"，将其替换为 argname
    - 在编译以下字段前调用：
        - 体热源 `ti_reyuan_expr` → argname = `"t"`
        - 材料 `daore_xishu` / `midu` / `bi_rerong` → argname = `"T"`
        - BC：`temperature` / `heat_flux` / `convection_coeff` / `T_inf` → argname = `"T"`
    - 字面替换后再 `expr::parse` 编译

- 错误模式：在字面替换前先扫一遍所有"孤立 x"位置
    - 对每个孤立 x，**不校验**——x 是合法 exprtk 变量名（body 内可有 x）
    - 函数名合法性：若 expr_str 中出现任何 `name(...)` 形态的 token（用 token 边界扫描
    抓取形如 `[A-Za-z_][A-Za-z0-9_]*\(...\)` 的子串），其前缀名必须在 `fns` 中
    已注册；否则 panic
        - 错误信息：`panic("unknown function 'foo' referenced in '...': must be declared in <Functions>")`

## 约束

- **自变量判定规则**（用户确认）：字符 `x` 的**前面和后面都不是字母或下划线**（即不是 `[A-Za-z_]`）
  即为"孤立 x"，需要替换
    - 字符串首（`i==0`）和字符串尾（`i+1==n`）视作非字母非下划线 → 边界 `x` 算孤立
    - 数字、运算符、括号、空格等非字母非下划线 → 不影响孤立判定
    - 字母 → 不算孤立
    - `_` 是标识符字符 → 视作"非孤立"（和字母同等处理）
    - 用户视角示例：
        - `test_gaussian(x)` 中的 `x`（前 `(` 后 `)`，都不是字母或下划线）→ 替换
        - `2*x` 中的 `x`（前 `2` 数字）→ 替换
        - `x*0.01+1` 中的 `x`（前 字符串首，后 `*`）→ 替换
        - `x_next` 中的 `x`（后 `_` 是下划线）→ **不替换**
        - `xx` 中的两个 `x`（前一个 `x` 的后是字母，后一个 `x` 的前是字母）→ **不替换**
        - `axb` 中的 `x`（前 `a` 后 `b`）→ **不替换**
- **自变量名无歧义**：用户的表达式中 `x` 即"待替换变量"。`T` / `t` 是后端 exprtk
  符号名（exprtk 编译时绑定到 `FieldContext::t` 槽）
- 实现方式：单次扫描字符串，记录每个孤立 `x` 位置，倒序替换为 argname
    - 倒序避免位置偏移
    - 不需要 std::regex（避免 ECMAScript 引擎对 lookbehind / `\b` 的支持问题）
- 替换前后都是 std::string
- 替换必须**先校验函数名已注册再替换 x**：函数名校验独立于 x 替换；同一遍扫描里同时记错误

## 验收

- `test_gaussian(x)` → `test_gaussian(t)`（在体热源中）
- `test_gaussian(x)` → `test_gaussian(T)`（在材料中）
- `2*x + x_next` 中：第一个 `x` 替换（前面是数字）、第二个 `x` 不替换（后面是 `_`）
- `xx + axb` 中所有 `x` 都不替换（前后都是字母或边界）
- 函数名 `test_gaussian` 等保持原样（不被重命名为 `test_gaussian_t` 之类）
- 引用未注册的函数名 → panic
- 单元测试覆盖：体热源（`t` 代入）、材料（`T` 代入）、BC（`T` 代入）、
  孤立 x 判定、未注册函数 panic

## 不做

- IO 解析（02）
- 注册 native（03）
