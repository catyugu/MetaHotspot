# ADR-0004: Expression Evaluation Split — Geometry vs Field/BC

## Status

Accepted

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

- The `expr` module exposes a `FieldExpression` type: compiled expression + symbol table. Evaluated by calling `eval(ctx)` with a `FieldContext`, returning a double.
- **Preprocessor compiles all expressions**: The preprocessor receives IO model structures containing raw expression strings and compiles them all into `FieldExpression` objects:
    - Material properties (k, rho, c) -> MaterialProps (each a FieldExpression)
    - BC parameters (T_dirichlet, q_neumann, h_cauchy, T_inf_cauchy) -> BCParamTable (each a FieldExpression)
    - Heat sources (Q from Block.ti_reyuan_expr) -> CellFields.heat_source (per-cell FieldExpression)
    - After preprocessing, no raw expression strings remain in the internal model.
- **Native functions**: In addition to string-based expressions, the `expr` module supports registering C++ functions directly via `register_native(name, func)`, where `func` is `std::function<double(const FieldContext&)>`. This handles cases that are awkward to express as strings: piecewise constant/linear functions over spatial domains, tabulated data, etc. Both exprtk-registered functions and native functions live in the expr module's pool and are resolved by name during `FieldExpression::from_string()`.
- `FieldExpression` also has a `make_constant(double)` factory for values that are just numbers — avoids the overhead of expression evaluation when the value is known at compile time.
