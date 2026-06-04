# ADR-0004: Expression Evaluation Split — Geometry vs Field/BC

## Status

Accepted. **Supersedes prior serialization approach** (2026-06): field/BC expression evaluation now uses `tbb::enumerable_thread_specific` for lockless parallel execution. Heat sources are stored as a deduplicated dictionary (`heat_source_table` + per-cell `uint16_t` index) instead of a per-cell `vector<CompiledExpression>`.

## Context

Expressions appear in two very different contexts in the XML:

1. **Geometry expressions**: `WidthExpression>w_top/2</WidthExpression>`, `ThicknessExpression>3</ThicknessExpression>` — evaluated once during preprocessing, variables are geometry constants like `w_top`, `h_middle`.
2. **Field/BC expressions**: `ConvectionCoefficient>10</ConvectionCoefficient>`, user-defined functions like `test_gaussian` — evaluated repeatedly during assembly (possibly thousands of times), context includes `{x, y, z, T, t}`.

## Decision

Keep two expression evaluation paths strictly separate:

### Geometry expressions

- Evaluated by a **thin geometry utility** (not the `expr` module) in `preprocessor`.
- All variables (e.g., `w_top`, `t_middle`) are resolved to concrete numbers before evaluation.
- Result: floating-point numbers used to construct the mesh geometry.
- Simple expression grammar: `+ - * / ( )` and numeric constants. No special functions needed.

### Field/BC expressions

- Handled by the **`expr` module** using `exprtk`.
- Context variables: `{x, y, z, T, t}` — spatial position, temperature, time.
- Material property expressions: `k(x,y,z,T,t)`, `ρ(x,y,z,T,t)`, `c(x,y,z,T,t)`.
- BC parameter expressions: `T_dirichlet(x,y,z)`, `q_flux(x,y,z,t)`, `h_convection(x,y,z,T,t)`.
- User-defined functions (e.g., `test_gaussian`) registered in a function pool.
- Variables referenced in field/BC expressions are **never** geometry variables — geometry is fixed at preprocessing time.

## Rationale

- Conflating geometry and field expressions would require passing geometry variable values into every assembly call — unnecessary overhead.
- Geometry expressions have a trivially simple grammar — no need for exprtk overhead.
- Field/BC expressions need exprtk's full power: trigonometric functions, exponentials, user-defined functions, variable symbol tables.
- Clear separation reduces accidental misuse and improves testability.

## Notes

- The `expr` module owns the expression runtime types: `FieldContext`, `FieldEvaluator`, and `CompiledExpression` are all defined in `src/expr/expr.hpp` under `mhs::expr`. `src/common/types.hpp` re-exports them to `mhs` so existing call sites in `internal_model.hpp` / `assembler` / `preprocessor` keep working with the shorter `mhs::CompiledExpression` name. The `CompiledExpression` type is a lightweight handle wrapping `shared_ptr<ExprTKCompiledTLS>`. The TLS wrapper holds a `tbb::enumerable_thread_specific<ExprTKCompiled>` so each thread sees its own private ExprTK AST. Evaluated by calling `eval(ctx)` with a `FieldContext`, returning a double. No mutex is taken on the eval hot path.
- **Preprocessor compiles all expressions**: The preprocessor receives IO model structures containing raw expression strings and compiles them all into `CompiledExpression` objects:
    - Material properties (k, rho, c) → `MaterialProps` (each a `CompiledExpression`)
    - BC parameters (T_dirichlet, q_neumann, h_cauchy, T_inf_cauchy) → `BCParamTable` (each a `CompiledExpression`)
    - Heat sources (Q from `Block.ti_reyuan_expr`) → `InternalModel::heat_source_table` (deduplicated dictionary), referenced per cell by `CellFields::heat_source_idx` (`std::vector<uint16_t>`). Index `0` is reserved for the default zero source.
    - After preprocessing, no raw expression strings remain in the internal model.
- **Native functions**: In addition to string-based expressions, the `expr` module supports registering C++ functions directly via `register_native(name, func)`, where `func` is `std::function<double(const FieldContext&)>`. This handles cases that are awkward to express as strings: piecewise constant/linear functions over spatial domains, tabulated data, etc. Both exprtk-registered functions and native functions live in the expr module's pool and are resolved by name during `expr::parse()`.
- `CompiledExpression` has two factories: `make_constant(double)` (no TLS, returns the value directly) and `make_evaluator(formula)` (allocates the `ExprTKCompiledTLS` handle). `expr::parse(formula)` auto-detects pure numeric literals and returns a constant; otherwise it does a one-shot trial compile on the main thread (to surface syntax errors early) and returns a `make_evaluator` handle.
- **Thread safety model**:
    - Registry mutations (`set_variable`, `register_native`, `register_function`, `clear_registry`) and `eval_geometry` are mutex-protected.
    - `parse()` is invoked on the main thread during preprocessing; it briefly takes the registry mutex to read the variable table.
    - `CompiledExpression::eval()` is lock-free. It calls `tls.local()` to obtain the calling thread's private `ExprTKCompiled`; the symbol slots `x_/y_/z_/T_/t_` are then mutated by that thread alone, so the TBB-parallel cell loop in the assembler never serializes.
    - The `tbb::enumerable_thread_specific` lazy-instantiates one AST per worker thread on first use; the formula string is captured by value in the constructor lambda, so no external lifetime dependency exists.
- **Heat source dictionary rationale**: when many cells share the same `ti_reyuan_expr` (a common case in layered chip stacks), per-cell `vector<CompiledExpression>` allocated N copies of the same AST. The dictionary (`heat_source_table` indexed by `uint16_t`) deduplicates by formula string and reduces the per-cell footprint from a heavy `CompiledExpression` (shared_ptr + ETS header) to a 2-byte index, while keeping the lockless `eval()` semantics intact.
