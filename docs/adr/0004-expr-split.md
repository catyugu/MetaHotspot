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

**Geometry.** Evaluated by `mhs::core::eval_geometry(formula, symbols)` with muparser against a per-call `SymbolTable` of pre-registered variables.

**Field / BC.** Handled by the `expr` module (muparser-backed). Context: `{x, y, z, T, t}`. Material laws, BC parameters, and per-block heat sources all flow through this path.

### Thread safety

- There is no global registry. Each `build_model()` call constructs a local `SymbolTable` (variables + native closures), threads it through every `parse()` / `eval_geometry()` call, and discards it on return. The setup path is single-threaded by construction.
- `parse()` runs during setup and independent calls may run concurrently with separate symbol tables; it constructs and validates the creating thread's `MuCompiled` through `tls.local()` and returns a `CompiledExpression` handle that reuses it. The `MuCompiledTLS` inside the handle holds a value-copy of the `SymbolTable`, so the handle is independent of any external state and is safe to share across threads.
- `CompiledExpression::eval()` uses per-thread parsers without an application-level mutex. Internally it holds a `shared_ptr<MuCompiledTLS>`, which wraps a `tbb::enumerable_thread_specific<std::unique_ptr<MuCompiled>>`. Each TBB worker thread lazily instantiates its own private muparser instance on first `tls.local()`; the formula string is captured by value and the ETS constructor lambda accesses the `SymbolTable` owned by its `MuCompiledTLS`; their lifetime is held by the shared handle. The `unique_ptr` element type keeps each AST's heap address stable so that the `NativeFnCtx` slots (registered with `DefineFunUserData`) keep their raw `FieldContext*` valid even when the ETS grows or the wrapper is copied. Each AST's `current_ctx_` field is written by the calling thread only on every `eval()`.
- Constant expressions (`make_constant`) short-circuit before touching the TLS.

### Heat source table

`Model::heat_source_table` is a `std::vector<CompiledExpression>` with one entry per Block, referenced by `SourceTerm::source`. Each volume projection evaluates its source at the piece centroid and multiplies by the overlap volume; several sources may contribute to one cell.

### Native functions

Native functions live in `SymbolTable::natives` — a `std::unordered_map<std::string, FieldEvaluator>` populated by the preprocessor via `mhs::sim::register_all_functions(symbols, definition.functions)`. At `parse()` time, every entry is bound into the compiled AST as a `DefineFunUserData` slot whose user data is a `NativeFnCtx { FieldEvaluator fe; const FieldContext* ctx_ptr; }` stored in a reserved vector owned by that `MuCompiled`. When muparser evaluates an expression like `fn(a, b)`, it first resolves each argument independently, then passes them as a raw `double*` + `int nargs` to the `native_fn_bridge` static, which forwards directly to the user's `FieldEvaluator` together with the current TLS `FieldContext*`. Because the `SymbolTable` is captured by value into the `MuCompiledTLS`, the expression engine needs no global registry. User-provided closures must themselves capture safe lifetimes and support concurrent calls.

## Rationale

- Geometry evaluation uses a separate muparser instance bound only to named variables; it requires no field context or native-function registration.
- Field/BC expressions need muparser's full power (trig, exp, user functions, symbol table).
- The TBB ETS pattern keeps the inner cell loop serialization-free, which matters because every nonlinear iteration re-evaluates every cell's materials, BC, and heat source.
- Block-level heat-source sharing avoids per-cell expression objects.

## Notes

- `FieldContext`, `FieldEvaluator`, `CompiledExpression` are **defined** in `src/numerics/expression/expr.hpp` under namespace `mhs::core`. The dependency arrow is `mhs::sim → mhs::core`, never the reverse.
- The runtime model exposes compiled handles; nonconstant handles retain their formula strings internally to construct each thread's parser.
