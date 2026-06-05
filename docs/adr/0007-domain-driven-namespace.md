# ADR-0007: Domain-Driven Namespace Layout

## Status

Accepted. Supersedes the implicit per-directory namespaces in src/.

## Context

The codebase grew organically with each subdirectory declaring its own `mhs::xxx` namespace (assembler, expr, io, nonlinear) — but postprocessor, scheduler, and solver leaked types directly into `mhs::` (the brand prefix). `preprocessor/` was split: the main `preprocessor.{hpp,cpp}` used `mhs` while sub-files used `mhs::preprocessor`. `common/` was inconsistent: most types sat flat in `mhs` while logger was promoted to `mhs::logger`. This produced three failure modes:

1. **Collision risk** — flat types in `mhs` (Preprocessor, Postprocessor, Scheduler, …) shared the same level as common POD types, so adding a new common type could shadow a subsystem class.
2. **Brand-prefix pollution** — `mhs` was supposed to be the library's brand identifier, but it also hosted concrete types, making the "brand" meaningless.
3. **Inconsistency cost** — readers had to look up which directory used which namespace, and the answer changed without warning.

## Decision

Adopt a **domain-driven, directory-decoupled** namespace scheme. Public API lives at exactly two levels `mhs::domain`; a third `mhs::domain::detail` exists only for cross-file internal implementation.

### Mapping

| Namespace     | Source directories                                               | Role                                                               |
| ------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| `mhs::core`   | `common/` (except `logger.*`) + `expr/`                          | Data models, expressions, POD enums, shared infrastructure         |
| `mhs::sim`    | `assembler/` `solver/` `scheduler/` `nonlinear/` `preprocessor/` | Numerical engine: assembly, linear/nonlinear solve, scheduling     |
| `mhs::io`     | `io/`                                                            | XML I/O, VTU output                                                |
| `mhs::post`   | `postprocessor/`                                                 | Cell-to-node interpolation, derived fields                         |
| `mhs::logger` | `common/logger.{hpp,cpp}`                                        | Standalone logging service (kept distinct, not folded into `core`) |

`mhs` itself is a **brand prefix** — it contains no type definitions, no `using` re-exports.

### Internal layer

- **Anonymous namespace** for symbols used in a single `.cpp` (modern replacement for `static`).
- `mhs::sim::detail` / `mhs::core::detail` / etc. for symbols shared across multiple `.cpp` files of the same domain but not part of the public API.

### `core` is the foundation

`mhs::core` is the only domain that may not include any sibling-domain header. When `core` needs behavior from `sim` (e.g., a "solve" function pointer), it declares an abstract interface or function-pointer typedef, and `sim` provides the concrete implementation. `Scheduler` (in `sim`) wires the two together.

### Flat inside `sim`

`mhs::sim` is **flat** — no third-level sub-namespace. The whole domain hosts ~20 symbols (Assembler, LinearSystem, LinearSolver, BiCGSTABSolver, PardisoSolver, SparseLUSolver, Scheduler, Preprocessor, NonLinearConfig, NonLinearResult, nonlinear_solve(), …) which is well within readable range. Putting each subclass of `LinearSolver` into its own sub-namespace would defeat the "everything solve-related in one place" goal.

### `Solver` → `LinearSolver`

The pre-existing class `Solver` in `src/solver/` is renamed to `LinearSolver` to disambiguate from the nonlinear iteration pathway. The nonlinear side keeps its free function `nonlinear_solve()` plus `NonLinearConfig` / `NonLinearResult` structs in `mhs::sim`.

### expr folded into core

`mhs::expr` is eliminated. `CompiledExpression`, `FieldEvaluator`, `FieldContext`, the registry, and the parse/eval functions all become `mhs::core::*`. `expr/` directory is a sub-organization of `mhs::core`, not its own domain. This reflects that expressions are a foundational service consumed by both `sim` and `io`, not a sibling domain.

### No header using-directives

`.hpp` files never contain `using namespace X;`. They always spell types out fully. `.cpp` files may use `using namespace ::tinyxml2;` or function-scope `using namespace exprtk;` where it improves readability, mirroring the current state of `src/io/io.cpp` and `src/expr/expr.cpp`.

### No re-exports under `mhs`

`src/common/types.hpp` currently re-exports expression types from `mhs::expr` into `mhs`. This re-export chain is **deleted**. Callers reference `mhs::core::CompiledExpression` directly. The brand namespace stays empty.

## Rationale

- **Two-level public API** matches the FHS/Abseil convention: short to read (`mhs::sim::LinearSolver`), deep enough to convey domain (`mhs` ≠ sim ≠ core).
- **Directory decoupling** lets us reorganize files (e.g., split a large `solver.cpp`) without churning namespace paths.
- **core is foundational** means that everyone reads the glossary of types from one place, and dependency cycles are physically impossible: a header in `mhs::core` cannot include a header from `mhs::sim` because the include would create a cycle (`sim` already depends on `core`).
- **Detail is the third level** — not sub-domains. Sub-domains would invite "I'll just add one more level" creep; `detail` is a single, named, recognizable concept.
- **logger stays separate** because it is a service, not a data model. Folding it into `core` would couple logging (a cross-cutting concern) to the data-layer type definitions, making logging headers drag in heavy types.
- **expr folded into core** because expressions are data that flows through the system, not a parallel computational domain. `mhs::expr` and `mhs::sim` being siblings implied they were parallel, but `expr` is upstream infrastructure.

## Migration

- One commit per domain rename; sequence is leaf-first to keep the build green between commits:
  1. `solver/` → `mhs::sim` (rename `Solver` to `LinearSolver`, hoist types).
  2. `assembler/` → `mhs::sim`.
  3. `nonlinear/` → `mhs::sim` (free function becomes `mhs::sim::nonlinear_solve`).
  4. `scheduler/` → `mhs::sim`.
  5. `preprocessor/` → `mhs::sim` (merge `mhs::preprocessor` and the main `preprocessor.{hpp,cpp}`).
  6. `expr/` → `mhs::core` (delete `mhs::expr`).
  7. `common/` (non-logger) → `mhs::core`.
  8. `postprocessor/` → `mhs::post`.
  9. Delete `mhs::` re-exports in `common/types.hpp`.
  10. Update `bin/main.cpp` and tests to use new qualified names.

Each step should compile and pass tests in isolation. The full migration is mechanical; the design work is the namespace map, captured above.

## Notes

- The brand namespace `mhs` becoming empty is intentional. It documents "this symbol belongs to the MetaHotspot library" without inviting the accumulation of types that motivated the cleanup.
- `mhs::logger` keeps its current shape — it already works and there is no pressure to fold it in.
- `io::read_xml` / `io::write_vtu` keep their `io::` prefix even though they live in `mhs::io` — `using` aliases in `bin/main.cpp` may shorten these to local names.
- Forward declarations must use the new qualified names: `mhs::core::InternalModel`, not `InternalModel`.
