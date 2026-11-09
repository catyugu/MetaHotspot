# Reduced components and physical connection maps

A reduced component is a meshless, linear thermal subsystem assembled alongside
Cartesian components. Its numerical formulation owns modal coordinates and reduced
matrices; its enclosing Component owns physical ports and exterior boundary
declarations. It has no fake internal Cartesian cells, material IDs or
cell-temperature field.

## State and operators

The supplied reduced system is

```text
C_r dz/dt + [K_r0 + Σ h_i K_ri] z = f_r0 + Σ h_i f_ri.
```

Matrices are dense row-major arrays of shape `(order, order)`; vectors have
length `order`. `StateLayout` assigns a separate `ReducedCoordinate` field.
The initial ROM coordinate vector is specified independently of
`Study.initial_temperature`, which initializes FOM temperatures.

An affine parameter term stores a name, a finite scalar coefficient and its
projected stiffness/load contributions. Compilation folds all affine terms into
the runtime K/f arrays, so parameter names and term matrices do not survive in
the compiled model. Recompiling the same ROM definition with different
coefficients therefore produces a new immutable snapshot without re-extracting
the ROM. Arbitrary time-varying coefficients are not yet part of the
reduced-domain contract.

## Physical thermal ports

Each ROM port has a physical, axis-aligned rectangular surface and a collection
of non-overlapping patches that exactly cover it. Each patch contains:

- `trace` and `offset`: adjacent physical cell-side temperature
  `T = trace * z + offset`;
- `half_resistance = d/k_n`, in m² K/W, for the distance between that
  temperature sampling point and the coupling plane;
- the patch's physical tangential center;
- optional modal maps of the two tangential temperature gradients, in K/m.

The two endpoints of a thermal connection share one physical patch flux.
For a matching patch of area `A`,

```text
g = A / (half_resistance_a + half_resistance_b)
q_a = g * (T_a - T_b),    q_b = -q_a.
```

The interface contribution is scattered through the transposes of the
physical trace maps (`P_a^T q_a`, `P_b^T q_b`). This is the Galerkin
coupling required for full-rank equivalence; modal coordinates are never
interpreted as physical temperatures.

For nonmatching **inter-domain** connections, the NCC tangential correction
is added to the same patch flux and scattered with the same signs.
A nonmatching ROM endpoint must supply both gradient maps on each
participating patch; missing maps cause a compile error. Merely using
common-patch TPFA without NCC is not an accepted substitute.
Local refinement **within one FOM** continues to use TPFA rather than NCC,
as specified by the generic-layout tracking document.

ROM ports and observations are lowered to runtime-only records before solve.
ROM observations map reduced coordinates to physical temperatures through
explicit `weights` and an optional offset. A named point probe can reference
such an observation; it does not read a nonexistent ROM cell.

## Public construction

Both C and Python clients can construct a ROM with base projected `K/C/f`,
optional initial coordinates, zero or more affine terms, rectangular ports,
and physical observation maps. ROM ports are submitted with all their patches
in one C ABI call, so failures cannot leave a partially constructed port. Connections use existing Domain/Port IDs and
the existing model/compiled/solution opaque handles.

Cartesian and reduced components share one creation-ordered component table.
The component ID is the stable index in that table, and the two formulations may be
interleaved. Models start empty; the first component, regardless of formulation,
receives ID 0. A reduced-only assembly therefore needs no placeholder Cartesian
component.

## Verification and limits

`tests/integration/test_rom_domain.cpp` covers full-rank projection against
an identical FOM (steady, transient, source, capacity, interface, probe),
ROM-only coupling, parametric HTC terms, and nonmatching NCC on a linear
interface trace. `python/tests/test_rom_domain.py` covers the public C/Python
path. The transient comparison uses an independent exact response to
distinguish adaptive time-step error from ROM algebraic error.

A reduced model must be supplied by an extractor or external source.
Automatic ROM basis extraction, full pressure–velocity coupling and
time-dependent affine parameter controls are outside the present contract. Those capabilities
must not be inferred from the existence of a reduced Component.
