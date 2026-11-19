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
surface's environment. FOMs use `facet_boundary` to select a thermal boundary group and its law. Instance overrides use `SurfaceId`. Contacts with
untrained EROM surfaces log a warning and leave both environment laws in effect. Fluid advection and
hydraulic boundary terms operate within the individual FOM.

An EROM starts at zero modal state when the study uses its reference
temperature. For a different uniform initial temperature, the solver scales
the stored uniform mode by the temperature difference. The caller may also
supply an explicit project initial state. FOM and EROM instances share the
same output timeline. `Solution::output_traces`
contains `OutputTrace` values keyed by component and output ID. Set the C++ option
`store_snapshot_states=false` to retain these traces and the final state without
storing every full state snapshot. The default preserves full histories.

## Native C++ extraction

Link the C++ caller against `mhs::erom_extraction` and supply its
`FomDefinition` and `MeshSpec`. The target is part of the default SDK build.
To build it without PTA, set `MHS_BUILD_EROM=ON` and `MHS_BUILD_APPS=OFF`.
The extractor compiles the FOM, trains and validates it, and can write the
result atomically as `.mherom`.

```cpp
#include "erom/extraction.hpp"

using namespace mhs::app::erom;
// definition and mesh come from the caller.
auto result = extract(definition, mesh, {{0, {5, 200}}, {1, {10, 500}}}, 1e-3);
write_erom("package.mherom", result.asset);
```

A group's ID is its index in `FomDefinition::thermal_boundaries`. The default
group has ID `thermal_boundaries.size()`. Every region of a selected group
must lie on the exterior of the FOM. Groups may cover full faces or parts of
faces. Groups that select interior faces or mix interior and exterior faces
are rejected.
EROM source, output and connectable group IDs are zero-based indices in their
asset tables. Extraction assigns these indices after selecting powered sources
and trained groups from the FOM.

Supply positive HTC limits in W/(m² K) for each training group. All compiled
groups with nonzero HTC must be selected. A selected adiabatic group becomes
connectable with zero default HTC. The original constant HTC and ambient are
used on exposed areas. Unselected surfaces retain outline geometry for contact
diagnostics.

The extractor accepts one nonempty solid FOM with constant positive material
properties and constant volumetric source densities. Each powered source group
is an independent power channel with a volume-weighted mean temperature output.
All point probes in the definition are retained using the FOM's Green–Gauss and
boundary-trace reconstruction. The extractor rejects probes whose
reconstruction depends directly on a replaceable boundary law. The asset
initially uses the FOM reference temperature and stores the uniform mode for
starting from other temperatures.

Dirichlet conditions, nonzero Neumann fluxes, nonlinear or nonconstant physics,
and regions with no HTC anchor are rejected during extraction. Dirichlet and
Neumann laws can be applied to a trained EROM surface in the receiving project.
Use `read_erom` to load the output into a project.

Extraction uses the Extended-FANTASTIC response-space algorithm. Physical
heat-source columns drive training. A thermally disconnected region with no
heat source uses its physical ambient-temperature inputs on selected HTC
surfaces, evaluated at each sampled HTC value. Global capacity-scaled spectral
estimates use a preconditioned low-end Rayleigh–Ritz iteration and restarted,
reorthogonalized high-end Krylov iteration. Both require a checked eigenpair
residual. Elliptic matching points are positive real shifts, evaluated with
arithmetic-geometric means using the complementary modulus. Each input keeps
its own response space. Ten successful log-uniform HTC residual probes advance
each input to the next shift. The fixed C++ random seed gives reproducible
native runs.

Each facet uses the exact TPFA conductance
`g_f(h) = A_f h / (1 + R_f h)`. Conductances are summed per owner cell for
full-order assembly. Reduced assembly uses the cached basis rows of those
cells. The same law supplies ambient loads and the endpoint operators for
spectral estimation. The application-local `ReusableAmg` builds at most one
lower-HTC hierarchy per active shift and shares it across all inputs and HTC
samples at that shift. CG checks its true relative residual using the current
matrix. Final compression uses centered, normalized responses, thin QR and a
small SVD, then preserves the uniform-temperature mode. Extraction statistics
include the number of response solves, AMG setups and CG iterations.

The one caller-supplied tolerance controls response residuals, basis
compression and sampled output acceptance. Output validation uses independent
AMG-CG FOM solves with a true relative residual at most the smaller of
`1e-8` and one-thousandth of the extraction tolerance. It checks several
settings in the training range and the default environment when anchored.
The measured error is the temperature difference divided by the larger of one
kelvin and the magnitude of the FOM temperature rise. Accuracy after embedding
depends on the attached geometry and loads.

## Native .mherom v3

The binary stream is little-endian and unpadded. Counts and IDs are unsigned
32-bit integers. Scalars and row-major matrix entries are IEEE 754 binary64.
Strings are UTF-8 bytes preceded by a 32-bit byte count.

| Order | Record | Layout |
| --- | --- | --- |
| 1 | Header | Eight bytes `MHEROM\r\n`, version `3`, then counts `n, ns, no, nb, ng, nl` |
| 2 | Temperature | Reference temperature |
| 3 | Operators | `K[n,n], C[n,n], B[n,ns], L[no,n], Vb[nb,n]` |
| 4 | Initialization | Uniform mode of length `n` |
| 5 | Interfaces | `nb` records of facet geometry, area-specific half-cell resistance and HTC group index |
| 6 | HTC groups | `ng` records of name and four scalars (HTC min, max, default and ambient temperature) |
| 7 | Outline | `nl` records of group name, facet count and facet geometries |
| 8 | Sources | `ns` records of name and default power in watts |
| 9 | Outputs | `no` records of name and source index. Point outputs use `0xffffffff` in place of the source index, followed by three coordinates |

Facet geometry contains axis (0/1/2), outward normal (0 for −, 1 for +),
plane coordinate and four tangent bounds. Interface records correspond
one-to-one to rows of `Vb`. Outline geometry is used for contact diagnostics.

The operators use `C dq/dt + Kq = BP + f_external`, with internal surface-side
temperature `T_ref + Vb q` and output temperature `T_ref + L q`. The native reader
checks the version, truncation, numeric values, dimensions and group indices.
Regenerate older assets with the current extractor before loading them.

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
