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

**Geometry.** Evaluated by `mhs::core::eval_geometry()` from a registry of pre-registered variables. All variables are resolved to concrete numbers *before* the expression runs. Grammar is `+ - * / ()` and numeric constants — no special functions.

**Field / BC.** Handled by the `expr` module (muparser-backed). Context: `{x, y, z, T, t}`. Material laws, BC parameters, and per-block heat sources all flow through this path.

### Thread safety

- Registry mutations (`set_variable`, `register_native`, `clear_registry`) and `eval_geometry` are mutex-protected.
- `parse()` is main-thread only; it briefly takes the registry mutex while doing a one-shot trial compile (to surface syntax errors early) and returns a `CompiledExpression` handle.
- `CompiledExpression::eval()` is **lock-free**. Internally it holds a `shared_ptr<MuCompiledTLS>`, which wraps a `tbb::enumerable_thread_specific<std::unique_ptr<MuCompiled>>`. Each TBB worker thread lazily instantiates its own private muparser instance on first `tls.local()`; the formula string is captured by value in the ETS constructor lambda, so there is no external lifetime dependency. The `unique_ptr` element type keeps each AST's heap address stable so that the `NativeFnCtx` slots (registered with `DefineFunUserData`) keep their raw `FieldContext*` valid even when the ETS grows or the wrapper is copied. Each AST's `current_ctx_` field is written by the calling thread only on every `eval()`.
- Constant expressions (`make_constant`) short-circuit before touching the TLS.

### Heat source dictionary

`InternalModel::heat_source_table` is a deduplicated `std::vector<CompiledExpression>` indexed by `CellFields::heat_source_idx` (`std::vector<uint16_t>`). Index 0 is reserved for the default `make_constant(0.0)`. Rationale: many cells share the same `ti_reyuan_expr` formula (common in layered chip stacks); per-cell `vector<CompiledExpression>` would allocate N copies of the same AST. The dictionary holds one AST per unique formula, reducing the per-cell footprint to 2 bytes, while keeping the lockless `eval()` semantics intact.

### Native functions

`register_native(name, func)` registers a `FieldEvaluator` — `std::function<double(const std::vector<double>& args, const FieldContext& ctx)>` — for cases awkward to express as strings (piecewise spatial, tabulated data). When muparser evaluates an expression like `fn(a, b)`, it first resolves each argument independently, then passes them as a raw `double*` + `int nargs` to the `native_fn_bridge` static, which packs them into `std::vector<double>` (`args`) and forwards to the user's `FieldEvaluator` together with the current TLS `FieldContext*`. They are bound into the parser via `DefineFunUserData()` (with a non-null `NativeFnCtx*` as user data) and resolved by name during `parse()`.

## Rationale

- Geometry expressions have trivial grammar — muparser overhead is unjustified.
- Field/BC expressions need muparser's full power (trig, exp, user functions, symbol table).
- The TBB ETS pattern keeps the inner cell loop serialization-free, which matters because every nonlinear iteration re-evaluates every cell's materials, BC, and heat source.
- Heat source deduplication is a memory win with zero semantic change.

## Notes

- `FieldContext`, `FieldEvaluator`, `CompiledExpression` are **defined** in `src/expr/expr.hpp` under namespace `mhs::core`. The dependency arrow is `mhs::sim → mhs::core`, never the reverse.
- After preprocessing, no raw expression strings remain in the internal model.
