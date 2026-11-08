# Compiled model, state layout and frozen flow

`ModelDefinition` owns global symbols, materials, sources, a default Study and named probes.
Its default `FomDomainDefinition domain` owns Cartesian mesh specifications,
volume geometry and thermal/fluid boundaries. This first refactor supports one
user FOM; creating additional domains and connecting them belongs to the next
stage. A fluid region is part of this FOM and is selected by the material's
`dynamic_viscosity` expression.

`build_model()` returns an independent `CompiledModel` containing a FOM domain
table, `StateLayout`, a Study snapshot and compiled probes. `FomDomain` owns immutable geometry,
material expressions, source quadrature, thermal boundary patches, fluid
membership and separately projected fluid boundary patches. Projection records
retain boundary subdivisions rather than collapsing them into mesh faces.
No mutable definition references survive compilation.

Each `FieldLayout` identifies a domain, quantity, support kind and component.
Its equally sized `entities` and `dofs` arrays map local entries independently to
support entities and global state entries. Cell, face and state indices are
separate contracts. Assembly and observation translate between local cell
fields and global DOFs; a temperature permutation preserves the solution.
The current frozen-flow solver allocates temperature unknowns only, including
both solid and fluid cells. It does not allocate unused pressure/velocity
unknowns. Those require the later flow-equation stage. ROM support and
connections are also later stages.

C clients query `state_count`, `domain_count`, `field_count`, domain/field
metadata and field mappings on the compiled handle. Python exposes the same
information as `Compiled.info`, `Compiled.domains`, `Compiled.field_layouts`.
These are copied snapshots with independent NumPy buffers. Geometry/material
outputs remain in domain-local cell order; state and operators use global DOF
order. The existing geometry copy function covers the single default FOM.

There is no model reference temperature. Each solve initializes its own frozen
hydraulic flow from the actual supplied state, or from
`Study.initial_temperature` when no state is supplied. Viscosity and density
expressions are evaluated at each fluid cell center and its initial temperature.
The pressure equation uses mass conductances and frozen interior/boundary mass
flows share one conservative balance. These flows remain fixed during the
thermal solve. A standalone operator assembly initializes its frozen flow from
the state/time passed to that assembly. It does not reuse another solve's flow.
Consequently inspecting operators later at a different state can describe a
different frozen-flow initialization for temperature-dependent fluid materials.

Actual inflow requires explicit temperature data. Pressure outlets accept a
backflow temperature through the same optional temperature argument as other
fluid boundaries. Missing temperature fails only if that boundary actually
admits fluid. Neither a model-level 300 K fallback nor the study initial
condition supplies a boundary temperature. PTA import keeps pressure-only
outlets valid for outflow; actual reversal without temperature reports an error.

All four C opaque handles remain independently owned. Destroying or changing a
model definition cannot change compiled snapshots; destroying a compiled handle
cannot change a solution or operator snapshot. No compatibility aliases or ABI
version mechanisms accompany these breaking changes.
