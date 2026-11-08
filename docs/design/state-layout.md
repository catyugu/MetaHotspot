# Compiled model, state layout and domain connections

`ModelDefinition` owns global symbols, materials, sources, a default Study and named probes.
Its `domains` table contains `FomDomainDefinition` objects, each owning Cartesian
mesh specifications, volume geometry, thermal/fluid boundaries and named ports.
Domain 0 exists on creation. Additional FOMs are independent computational
components. A fluid region remains part of its FOM, selected by the material's
`dynamic_viscosity` expression.

`build_model()` returns an independent `CompiledModel` containing FOM domains,
`StateLayout`, a Study snapshot, probes and compiled thermal connections.
`FomDomain` owns immutable geometry, material expressions, source quadrature,
thermal boundary patches, fluid membership and fluid boundary projections.
Projection records retain subdivisions rather than collapsing them into mesh faces.
No mutable definition references survive compilation.

Each `FieldLayout` identifies a domain and physical quantity and owns one
contiguous global DOF range `[offset, offset + count)`. Support kind is derived
from the quantity. Physical support identity remains in the compiled domain:
temperature uses all FOM cells in CellId order, pressure uses
`fluid_to_global`, velocity uses `velocity_faces`, and ROM coordinates use
their natural modal order. The layout therefore does not duplicate entity and
DOF arrays or support arbitrary post-compilation DOF permutations. A Stokes FOM
has one temperature field, one pressure field and one normal-face velocity
field; face orientation comes from the mesh face itself.

C clients query `state_count`, `domain_count`, `field_count`, domain/field
metadata, contiguous DOF ranges and support entities on the compiled handle. Python exposes copied snapshots
as `Compiled.info`, `Compiled.domains`, `Compiled.field_layouts`.
Geometry, evaluated materials and source projections concatenate domains in
DomainId order. Exported geometry cell references include that domain's cell
offset; field support-entity IDs remain domain-local. State and operators use
the canonical contiguous global DOF ranges. Source IDs retain their model-global identity.

## Thermal connections

A port is a named SI rectangle on exterior solid cells of a FOM. Connections
reference pairs of `(DomainId, PortId)`. Each declared port must be connected
exactly once to a distinct domain; surfaces must coincide with opposing normals
and complete coverage. Overlapping ports and intersections with explicit thermal
or fluid conditions are errors. Ports replace the default external boundary on
exactly their integration region, retaining subdivisions of the remaining face.
Connections through fluid interface cells are explicitly unsupported in this
stage; a connected FOM may still contain an internal fluid channel.

A matching interface uses the same normal series resistance as an interior
solid face. Nonmatching interfaces are intersected into common rectangles. For
one rectangle of area `A`, normal half-distances `d_A,d_B`, diagonal normal
conductivities `k_A,k_B`, and tangential offsets from cell centers to its centroid:

```
g = A / (d_A/k_A + d_B/k_B)
q_A = g * (T_A - T_B + delta_A.grad_t(T_A) - delta_B.grad_t(T_B))
q_B = -q_A
```

Tangential gradients use area-weighted directional differences over internal
neighbours and physical boundary traces. This reproduces linear fields on the
supported Cartesian grids, including one-sided physical-boundary stencils.
The two-point term stays implicit. NCC offsets enter opposite load entries and
are refreshed at every nonlinear iterate, also within every transient stage.
The existing nonlinear residual checks therefore include the corrected flux;
failed corrector convergence propagates as a failed solve. No public switch can
silently disable NCC. Solver DEBUG logs report the iterate's full equation
residual and update. `connection_flux()` evaluates the same flux kernel for an
arbitrary supplied state/time and does not initialize a new hydraulic solve.

Physical probes receive the eliminated common interface temperature when
reconstructing cell gradients. This preserves near-interface observations under
matching decomposition, as well as cell-centered states. A probe shared by two
domains is ambiguous and reports an error; select a point inside one component.

Python construction uses integer IDs, without additional owning handles:

```python
upper = model.add_fom_domain()
model.set_cartesian_mesh(x, y, upper_z, domain=upper)
volume = model.add_volume(copper, domain=upper)
model.add_box(volume, upper_min, upper_max, domain=upper)
a = model.add_port(0, "top", interface)
b = model.add_port(upper, "bottom", interface)
connection = model.connect(0, a, upper, b)
compiled = model.compile()
heat_rates = compiled.connection_flux(connection, compiled.solve().state)
```

C domain-local construction functions take `domain` immediately after the model
pointer. Python supplies `domain=0` by default. C++ callers use `domains[0]` for
the default FOM. These are breaking C/C++ changes, with no compatibility aliases.

Within a FOM, internal faces continue to use TPFA without NCC. The current public
Cartesian mesh specification does not yet express local octree refinement;
GL-06 remains pending that capability. Connections currently implement perfect
solid thermal contact; ROM ports and coupled flow connections remain later work.

## Frozen flow and ownership

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
backflow temperature through the optional temperature argument. Missing data
fails only if that boundary actually admits fluid. Neither a model-level 300 K
fallback nor the study initial condition supplies a boundary temperature. PTA
import keeps pressure-only outlets valid for outflow; actual reversal without
temperature reports an error.

All four C opaque handles remain independently owned. Destroying or changing a
model definition cannot change compiled snapshots; destroying a compiled handle
cannot change a solution or operator snapshot. No ABI version mechanisms are
introduced.
