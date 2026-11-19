# Mixed projects and EROM extraction

The C++ SDK compiles projects from reusable FOM and EROM assets. Each component
has its own ID, translation and optional surface laws. An EROM instance can
also set a time-dependent power in watts for each source ID. FOM source and
material expressions use the FOM's local coordinates. Surface laws use the
local quadrature point on each exposed patch.

```cpp
#include "compiler/project_compiler.hpp"
#include "solver/solver.hpp"

using namespace mhs::model;

auto package = read_erom("package.mherom");
ProjectDefinition input;
input.components.push_back({10, {0, 0, 0}, package});
input.components.push_back({20, {0.1, 0, 0}, package});
input.components[1].powers[3] = [](double t) { return t < 0.1 ? 5.0 : 2.0; };
auto project = mhs::compiler::compile_project(std::move(input));
auto solution = mhs::solver::solve(project, {mhs::solver::StudyType::Transient, 300, 2, 0.1});
const auto* trace = solution.find_output(20, 0);
```

Build a FOM with `mhs::compiler::compile_fom` and share the result as
`std::shared_ptr<const Fom>`. `Project::dof_offset` and `dof_size` identify a
component's block in each state snapshot. FOM blocks store absolute cell
temperatures. EROM blocks store modal states. `output_position` translates
point observations into world coordinates.

The compiler locates opposing coplanar facets and couples their overlap by TPFA.
`ProjectDefinition::contact_resistance` accepts area-specific contact
resistances in m² K/W keyed by ascending component IDs. A connected patch
replaces the environment over its covered area. The remaining area keeps the
surface's environment. Geometric surface identity (`facet_surface`) is independent
of FOM thermal-condition IDs (`facet_thermal_condition`). Instance overrides use
SurfaceId. Contacts with untrained EROM surfaces generate a
warning and leave the environment on both sides in effect. Fluid advection and
hydraulic boundary terms operate within the individual FOM.

The EROM asset stores an initial modal state and may include the mode response
to a uniform change in initial temperature. If this response is absent, a
transient study with a different initial temperature requires an explicit
initial modal state. FOM and EROM instances share the same output timeline. `Solution::output_traces`
contains `OutputTrace` values keyed by component and output ID. Set the C++ option
`store_snapshot_states=false` to retain these traces and the final state without
storing every full state snapshot. The default preserves full histories.

## Standalone PTA extraction

Build with `MHS_BUILD_APPS=ON`. The standalone C++ executable uses the same PTA
XML reader as the simulation CLI. Model preparation, extraction, validation and
atomic `.mherom` writing run in one process, with no Python runtime or temporary
operator exchange file.

```sh
metahotspot-erom model.xml --list-surfaces
metahotspot-erom model.xml package.mherom --htc 0 5 200 --htc 1 10 500 \
  --probe junction
```

The native SDK installs `metahotspot-erom` in `bin/`. Each `--htc`
argument gives a surface ID and positive lower and upper HTC limits in
W/(m² K). All surfaces with nonzero ambient HTC must be selected. A selected
adiabatic surface becomes connectable with a zero default environment HTC.
The original constant HTC and ambient are the defaults on exposed areas.

The extractor currently accepts one nonempty solid FOM with constant positive
material properties and constant volumetric source densities. Each powered
block is an independent power channel. PTA block names label those channels
when present. Source temperature means the volume-weighted cell temperature
of its source region. Point temperatures use the FOM's Green–Gauss and
boundary-trace reconstruction. The extractor rejects selected probes whose
reconstruction depends directly on a replaceable boundary law. Use repeatable
`--probe` to select named probes, `--no-point-probes` to omit them, or
`--no-source-temperatures` to omit source means.

Dirichlet conditions, nonzero Neumann fluxes, nonlinear or nonconstant physics,
and regions with no HTC anchor are rejected during extraction. Dirichlet and
Neumann laws can be applied to a trained EROM surface in the receiving project.
Use the C++ SDK to load and run the output. Existing PTA applications continue
to handle their usual XML studies.

Extraction uses the Extended-FANTASTIC response-space algorithm, implemented
privately in `apps/metahotspot-erom`. By default the physical heat-source columns
drive training. A thermally disconnected region with no heat source uses its physical
ambient-temperature inputs on selected HTC surfaces, evaluated at the HTC lower
bounds, so passive dynamics are retained. No additional boundary enrichment is
performed. Global
capacity-scaled spectral estimates use a preconditioned low-end Rayleigh–Ritz
iteration and restarted, reorthogonalized high-end Krylov iteration. Both require
a checked eigenpair residual. Elliptic matching points are positive real shifts,
evaluated with arithmetic-geometric means using the complementary modulus.
Each source keeps its own response space and incremental Galerkin matrices.
Ten successful log-uniform HTC residual probes advance each source to the next
shift. The fixed C++ random seed gives reproducible native runs; sampling and
basis coordinates may differ from the former Python implementation.

The application-local `ReusableAmg` owns an AMGCL smoothed-aggregation/SPAI0
hierarchy separately from CG. At most one lower-HTC hierarchy is built per active
shift and shared by all ports, HTC samples and response warm starts at that shift.
CG receives the current matrix explicitly; each solve checks its true relative
residual. The solid-only extraction operators are symmetric positive definite
after training HTC closure. Final compression uses centered, normalized responses,
thin QR and a small SVD, then preserves the uniform-temperature mode.
`--max-order` bounds response solves before compression. The CLI reports response solves,
response AMG setups and CG iterations.

Each training group uses an affine approximation between the exact
endpoint TPFA conductances. The extractor validates temperature outputs by
solving the FOM with AMG-CG (true relative residual at most the smaller of `1e-8`
and `--output-error / 1000`) and exact TPFA conductances at several settings in
the training range and at the default environment when anchored. `--output-error`
bounds the sampled error divided by the larger of one kelvin and the magnitude
of the reference temperature rise. The output reports the measured maximum.
Accuracy after embedding depends on the attached geometry and loads.

## Native .mherom v1

The binary stream is little-endian and unpadded. Counts and IDs are unsigned
32-bit integers. Scalars and row-major matrix entries are IEEE 754 binary64.
Strings are UTF-8 bytes preceded by a 32-bit byte count.

| Order | Record | Layout |
| --- | --- | --- |
| 1 | Header | Eight bytes `MHEROM\r\n`, version `1`, then counts `n, ns, no, nb, nf, ng` |
| 2 | Temperature and training | Reference and initial temperature, training tolerance, uniform-initialization flag |
| 3 | Operators | `K[n,n], C[n,n], B[n,ns], L[no,n], Vb[nb,n], R[nb]` |
| 4 | Initialization | Modal initial state of length `n`, followed by length-`n` uniform mode if the flag is `1` |
| 5 | Facets | `nf` records of axis (0/1/2), outward normal (0 for −, 1 for +), plane, and four tangent bounds |
| 6 | Surfaces | `ng` records of ID, name, trained flag, four scalars if trained (HTC min/max/default and ambient), facet count, and facet IDs |
| 7 | Sources | `ns` records of ID, name and default power in watts |
| 8 | Outputs | `no` records of ID, name, kind (0 for point, 1 for source average), temperature offset, and either three coordinates or the source ID |

`n` is the modal order. The first `nb` facets correspond one-to-one to the
rows of `Vb` and the positive area-specific half-cell resistances `R`. The
remaining facets describe the outline for diagnostic contact detection. Every
facet belongs to one surface. All HTC groups index the same `Vb` and `R`. C++ stores their training ranges and
default environment as `EromSurfaceTraining` in `Erom::surface_training`.
The operators use `C dq/dt + Kq = BP + f_external`, with internal surface-side
temperature `T_ref + Vb q` and output temperature `T_offset + L q`. The format
stores no mesh or full-field basis. The native reader checks the version,
truncation, numeric values, dimensions and facet ownership.

FloTHERM exports use a separate format. See
[`bci_rom_testcase1`](../playground/bci_rom_testcase1/) and the archived
[`simple_erom_case1`](../playground/archived/simple_erom_case1/) for comparisons.

The native extractor and writer live only in `apps/metahotspot-erom`. Python
`macromodel` utilities remain available for independent research workflows.
Component coupling uses the C++ Project API.

## Assembly measurements

Synthetic comparison against `dev` commit `350e8a2`, GCC 14.2 Release, two
OpenMP threads. Times are medians of seven workspace constructions; each
workspace performs twenty assemblies. The FOM pair has 40³ cells per cube and
1,600 contact patches. The EROM pair has 48 modes per asset and 400 patches.

| Operation | Before | After |
| --- | ---: | ---: |
| 128,000-DOF FOM pair: workspace setup | 663 ms | 18.9 ms |
| FOM pair: one assembly | 10.3 ms | 10.5 ms |
| 96-DOF EROM pair: workspace setup | 0.141 ms | 1.71 ms |
| EROM pair: one assembly | 1.83 ms | 0.0284 ms |

The EROM cache moves constant contact work into setup and adds one CSR-sized
coefficient array plus one load vector for reduced-only projects. Mixed
projects keep dynamic FOM resistance evaluation without this cache. CSR
structures match the baseline exactly; all K/C/load entries agree within
`rtol=atol=1e-12`. These measurements cover assembly, not extraction or complete
simulation time.
