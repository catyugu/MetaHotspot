# Generic model refactor work log

> **TEMPORARY MIGRATION DOCUMENT — DELETE BEFORE MERGING THIS PR.**
>
> This file exists only to keep the refactor resumable while the branch is in progress.
> It must not be present in the final merge commit.

## Goal

Decouple the simulation kernel and public SDK/ABI from the PTA front-end XML schema and presentation/output formats.

Target flow:

```text
PTA XML
  -> apps/metahotspot PTA compatibility adapter
  -> model::ModelDefinition
  -> compiler::build_model()
  -> model::Model
  -> solver::solve()
  -> solver::Solution
  -> apps/metahotspot presentation/export adapters
```

The library must consume a generic physical model. It must not know that a model originated from PTA XML.

## Required boundary

### Application layer owns

- PTA XML parsing and result XML write-back.
- PTA layer/block/rectangle semantics.
- PTA length units and conversion to SI.
- PTA geometry-expression evaluation needed to flatten layer/block offsets and thicknesses.
- PTA expression-argument rewriting. In particular, conversion of the PTA convention that uses `x` as a generic formal argument into canonical kernel variables such as `T` or `t`.
- Node-centered sampling required only by PTA result XML.
- VTU serialization and any connectivity constructed only for export.
- Front-end-specific labels and schema details.

### Library owns

- A generic `ModelDefinition` expressed in SI.
- Generic axis-aligned volume geometry and ordered add/subtract operations.
- Materials, source laws, thermal/fluid boundary definitions, probes and study settings.
- Compilation of canonical expressions without front-end rewrites.
- Discrete runtime model and solver.
- Raw solution/state access through the C ABI/Python wrapper.

## ModelDefinition target

Remove public input concepts that exist only because of PTA:

- `LengthUnit`
- `LayerSpec`
- `BlockSpec`
- 2-D PTA `RectSpec` as the primary body description

Replace them with generic 3-D volume definitions in SI. Initial implementation remains deliberately restricted to axis-aligned box CSG because the solver uses Cartesian hexahedral cells.

Material/source references should use IDs inside the generic definition where practical; name resolution belongs at adapters/builders.

## Output cleanup

Remove presentation-only state from the runtime model:

- `OutputMesh`
- `Model::output`
- output-only point/cell connectivity
- solver-owned node interpolation

Delete the library-side `mhs_io` target once its remaining responsibilities have moved to the application.

The C ABI and Python API must not expose:

- XML readers/writers
- layer/block/rect construction
- VTU writing

They should expose generic model construction plus raw compiled/solution data.

## Expression rule

The compiler accepts canonical expressions only.

Examples:

- material property law: canonical dependence on `T`
- source law: canonical dependence on `t`, and spatial coordinates where applicable
- boundary law: canonical dependence on `T` and other supported field variables

No `substitute_function_args(...)`-style front-end compatibility rewrite is allowed in the compiler. PTA-specific rewriting is performed before the `ModelDefinition` reaches the library.

## Implementation sequence

- [x] Create refactor branch and temporary work log.
- [ ] Replace layer/block geometry in `ModelDefinition` with generic SI volume geometry.
- [ ] Rewrite geometry compilation to consume generic volumes directly.
- [ ] Remove compiler-side expression argument rewriting.
- [ ] Move PTA XML reader and compatibility transforms into `apps/metahotspot`.
- [ ] Move PTA XML result writing, node sampling and VTU output into `apps/metahotspot`.
- [ ] Remove `OutputMesh` and output-only runtime state.
- [ ] Remove `mhs_io` from the library dependency graph.
- [ ] Replace PTA-shaped C ABI/Python construction API with generic volume/source construction.
- [ ] Update existing examples/tests only as required by the API change; do not add new tests unless necessary.
- [ ] Run build/test checks and repair regressions.
- [ ] Delete this file before opening/finalizing the PR.
- [ ] Open PR to `dev`.

## Design constraints

- Do not generalize the internal body geometry beyond what the current Cartesian/hexahedral solver needs.
- Keep `GeometryOperation::Add/Subtract` if useful; CSG operation semantics are generic.
- Do not introduce an assembly/subassembly/part object hierarchy merely to imitate FloTHERM UI organization.
- Keep the compiler input format independent of any one front end.
- Avoid compatibility shims that preserve PTA concepts in the public ABI.
- Prefer deleting obsolete abstractions over wrapping them in another layer.
