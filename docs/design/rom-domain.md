# Thermal ROM domains and physical connection maps

A ROM domain is a meshless, linear reduced thermal subsystem assembled alongside
full-order Cartesian domains. It owns modal coordinates and reduced matrices;
its physical interface is represented explicitly by boundary-port maps. A ROM
has no fake internal FOM cells, material IDs or cell-temperature field.

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
projected stiffness/load contributions. Each immutable compiled snapshot fixes
these coefficients. A caller can compile another model using the **same ROM
matrices and port maps** with different coefficients; this does not involve
re-extracting the ROM. Arbitrary time-varying coefficients are not yet part of
the reduced-domain contract.

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

ROM observations map reduced coordinates to physical temperatures through
explicit `weights` and an optional offset. A named point probe can reference
such an observation; it does not read a nonexistent ROM cell.

## Public construction

Both C and Python clients can construct a ROM with base projected `K/C/f`,
optional initial coordinates, zero or more affine terms, rectangular ports,
and physical observation maps. ROM ports are submitted with all their patches
in one C ABI call, so failures cannot leave a partially constructed port.
Connections use the same Domain/Port IDs and model/compiled/solution opaque
handles as FOM connections.

Domain IDs follow creation order, including interleaved FOM and ROM domains.
The typed domain registry maps each stable Domain ID to the corresponding
local FOM or ROM storage index. A ROM-only assembly is valid; it does not
instantiate a default empty FOM.

## Verification and limits

`tests/integration/test_rom_domain.cpp` covers full-rank projection against
an identical FOM (steady, transient, source, capacity, interface, probe),
ROM-only coupling, parametric HTC terms, and nonmatching NCC on a linear
interface trace. `python/tests/test_rom_domain.py` covers the public C/Python
path. The transient comparison uses an independent exact response to
distinguish adaptive time-step error from ROM algebraic error.

A reduced model must be supplied by an extractor or external source.
Automatic ROM basis extraction, full pressure–velocity coupling and
time-dependent affine parameter controls are outside PR3. Those capabilities
must not be inferred from the existence of a ROM Domain.
