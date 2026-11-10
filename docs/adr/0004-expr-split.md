# ADR-0004: Runtime expression evaluation

## Status

Accepted.

## Context

Materials, thermal boundary laws, and volumetric heat sources depend on spatial coordinates, temperature, and time. They are evaluated during assembly and may be shared across threads.

PTA layer, block, rectangle, and probe coordinate expressions are input-format details. The PTA adapter resolves them into numerical SI coordinates before constructing the SDK model definition.

## Decision

The runtime expression system accepts field expressions in the canonical context \`x/y/z/T/t\` and supports registered native functions. It has no geometry variables or geometry-evaluation entry point.

\`ModelDefinition\` holds numerical geometry and a table of runtime function specifications. The compiler builds a local \`SymbolTable\` containing only native function bindings; it parses material laws, heat sources, and thermal boundary laws into \`CompiledExpression\` handles.

PTA geometry arithmetic is handled entirely in \`apps/metahotspot/pta\` by a small evaluator, independently of the runtime expression system. Variables are resolved in input order, each referring only to previously defined variables. The evaluator supports numeric literals, identifiers, parentheses, unary signs, \`+ - * / ^\`, and yields a \`double\`. Geometry is converted to SI units by the PTA adapter.

### Thread safety

- No global registry is used. Each \`build_model()\` constructs and passes a local native-function symbol table.
- \`parse()\` validates the formula by constructing the creating thread's \`MuCompiled\` and returns a \`CompiledExpression\` that reuses this parser.
- Each nonconstant expression contains a \`shared_ptr<MuCompiledTLS>\` with a value-copy of the native-function table. A TBB \`enumerable_thread_specific<std::unique_ptr<MuCompiled>>\` gives each evaluating thread a private parser.
- \`MuCompiled\` keeps its \`FieldContext\` and the native-function callback slots at stable addresses; the parser updates the context before each evaluation.
- Constant expressions short-circuit without accessing thread-local storage.

### Heat source table

\`Model::heat_source_table\` holds one compiled expression per source definition. \`SourceTerm::source\` references this table. Source terms are evaluated at their physical integration points and multiplied by the subvolume weight.

### Native functions

\`SymbolTable::natives\` contains callbacks and their temperature-dependency metadata. The compiler registers \`ModelDefinition::functions\` using \`register_all_functions()\`. Muparser evaluates callback arguments, then calls \`native_fn_bridge\` with the bound \`NativeFnCtx\` and the active \`FieldContext\`.

## Rationale

Runtime expressions require reusable, thread-safe evaluation against the evolving physical state. PTA geometry arithmetic is an input-normalization operation and has no role in this runtime mechanism. Keeping them independent removes geometry variables from the public model contract and avoids compiling runtime functions during PTA geometry normalization.
