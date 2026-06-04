# ADR-0004: Expression Evaluation — Geometry vs Field/BC Split, with Lockless Field Eval

## Status

Accepted.

## Context

XML expressions come in two flavours:

1. **Geometry**: `w_top/2`, `thickness_expr` — evaluated **once** at preprocessing. Context: named variables.
2. **Field / BC**: `k(x,y,z,T,t)`, BC parameters, user-defined functions — evaluated **per cell per iteration**. Context: `{x, y, z, T, t}`.

Conflating them would force geometry constants into every assembly call.

## Decision

Two separate paths.

**Geometry.** Evaluated by `mhs::expr::eval_geometry()` from a registry of pre-registered variables. All variables are resolved to concrete numbers *before* the expression runs. Grammar is `+ - * / ()` and numeric constants — no special functions.

**Field / BC.** Handled by the `expr` module (exprtk-backed). Context: `{x, y, z, T, t}`. Material laws, BC parameters, and per-block heat sources all flow through this path.

### Thread safety

- Registry mutations (`set_variable`, `register_native`, `register_function`, `clear_registry`) and `eval_geometry` are mutex-protected.
- `parse()` is main-thread only; it briefly takes the registry mutex while doing a one-shot trial compile (to surface syntax errors early) and returns a `CompiledExpression` handle.
- `CompiledExpression::eval()` is **lock-free**. Internally it holds a `shared_ptr<ExprTKCompiledTLS>`, which wraps a `tbb::enumerable_thread_specific<ExprTKCompiled>`. Each TBB worker thread lazily instantiates its own private ExprTK AST on first `tls.local()`; the formula string is captured by value in the ETS constructor lambda, so there is no external lifetime dependency. Each AST's `x_/y_/z_/T_/t_` slots are written by the calling thread only.
- Constant expressions (`make_constant`) short-circuit before touching the TLS.

### Heat source dictionary

`InternalModel::heat_source_table` is a deduplicated `std::vector<CompiledExpression>` indexed by `CellFields::heat_source_idx` (`std::vector<uint16_t>`). Index 0 is reserved for the default `make_constant(0.0)`. Rationale: many cells share the same `ti_reyuan_expr` formula (common in layered chip stacks); per-cell `vector<CompiledExpression>` would allocate N copies of the same AST. The dictionary holds one AST per unique formula, reducing the per-cell footprint to 2 bytes, while keeping the lockless `eval()` semantics intact.

### Native functions

`register_native(name, func)` registers a `std::function<double(const FieldContext&)>` for cases awkward to express as strings (piecewise spatial, tabulated data). They live in the expr pool alongside exprtk-registered functions and are resolved by name during `parse()`.

## Rationale

- Geometry expressions have trivial grammar — exprtk overhead is unjustified.
- Field/BC expressions need exprtk's full power (trig, exp, user functions, symbol table).
- The TBB ETS pattern keeps the inner cell loop serialization-free, which matters because every nonlinear iteration re-evaluates every cell's materials, BC, and heat source.
- Heat source deduplication is a memory win with zero semantic change.

## Notes

- `FieldContext`, `FieldEvaluator`, `CompiledExpression` are **defined** in `src/expr/expr.hpp` (namespace `mhs::expr`); `src/common/types.hpp` re-exports them as `mhs::FieldContext` etc. so internal_model / assembler / preprocessor keep using the short name without including `expr/` directly. The dependency arrow is `common → expr`, never the reverse.
- After preprocessing, no raw expression strings remain in the internal model.
