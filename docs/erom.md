# Mixed projects and EROM extraction

The C++ SDK compiles projects from reusable FOM and EROM assets. Each component
is identified by its array index and stores a translation and optional surface laws. An EROM instance can
also set a time-dependent power in watts for each source ID; omitted schedules supply zero power. FOM source and
material expressions use the FOM's local coordinates. Surface laws use the
local quadrature point on each exposed patch.

```cpp
#include "compiler/project_compiler.hpp"
#include "solver/solver.hpp"

using namespace mhs::model;

auto package = read_erom("package.xml");
ProjectDefinition input;
input.components.push_back({{0, 0, 0}, package});
input.components.push_back({{0.1, 0, 0}, package});
input.components[1].powers.resize(4);
input.components[1].powers[3] = [](double t) { return t < 0.1 ? 5.0 : 2.0; };
auto project = mhs::compiler::compile_project(std::move(input));
auto solution = mhs::solver::solve(project, {mhs::solver::StudyType::Transient, 300, 2, 0.1});
const auto* trace = solution.find_output(1, 0);
```

Build a FOM with `mhs::compiler::compile_fom` and share the result as
`std::shared_ptr<const Fom>`. `Project::dof_offset` and `dof_size` identify a
component's block in each state snapshot. FOM blocks store absolute cell
temperatures. EROM blocks store modal states. `output_position` translates
point observations into world coordinates.

The compiler locates opposing coplanar facets and couples their overlap by TPFA.
`ProjectDefinition::contact_resistance` accepts area-specific contact
resistances in m² K/W keyed by ascending component array indices. A `Contact`
stores the matched area and identifies both sides by component and face index
(FOM facet or EROM interface). `Project::face_patches` provides one sequence of
component-local rectangles for each partitioned face, with a contact index on
each connected patch. A connected patch
replaces the environment over its covered area. The remaining area keeps the
surface's environment. Each FOM exterior facet carries its thermal boundary group index. Instance overrides use `BoundaryGroupId`. Contacts with
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

## Standalone PTA extraction

```sh
metahotspot-erom --input project.xml --output package.xml
```

Build/install the executable with `MHS_BUILD_EROM=ON`; it is independent of
`MHS_BUILD_APPS`. The input is the ordinary PTA `Structure` XML document,
augmented with one extraction configuration. No `StudyType`, transient study
duration, precomputed simulation or separate geometry representation is
required. The shared PTA reader evaluates geometry and converts lengths to SI.
`GeneratedMesh` (three axis coordinate arrays) takes precedence over the
legacy `Results/a:anyType/Mesh` when present and not nil. Mesh sources are not
combined.

```xml
<EromExtraction version="1">
    <Tolerance>1e-3</Tolerance>
    <BoundaryGroups>
        <BoundaryGroup index="0">
            <HtcMin>1</HtcMin>
            <HtcMax>10000</HtcMax>
        </BoundaryGroup>
    </BoundaryGroups>
</EromExtraction>
```

`Tolerance` is optional (default 1e-3), finite and strictly between zero and
one. At least one boundary group is required. `index` is the zero-based
position of the original `Boundaries/Boundary` element, not the compiled
group ID. References must be unique and identify explicit convection groups.
HTC limits are finite positive SI W/(m² K), with max >= min. Every original
region must have positive area, be covered by actual exterior facets and lie
on one of the six planes of the active model AABB. A group may span multiple
planes. Unselected groups (including recessed faces) remain fixed, not
connectable.

Explicit sources are declared in `HeatSources/HeatSource` using `Id` and
`Name` and referenced by each `Block/HeatSourceId`. Source declarations
determine channel order; multiple blocks sharing an ID share one channel.
PowerDensity and legacy VolumetricHeatSource expressions for referenced
sources are not evaluated or stored. A block without a source ID may have an
absent or literal-zero legacy expression; other expressions are rejected to
avoid silently discarding a source. Ordinary PTA simulations retain the
legacy VolumetricHeatSource behavior. The CLI returns 0 on success, nonzero
on failure; diagnostics go to stderr and successful extraction prints order,
measured output error and elapsed time. Output replacement is atomic.

## Native C++ extraction

Link against `mhs::erom_extraction` to supply a `FomDefinition`, `MeshSpec`,
HTC ranges and tolerance directly. A selected group's ID is its index in
`thermal_boundaries` (the implicit default group cannot be selected).
All `FomDefinition::sources` referenced by active cells are explicit channels.

```cpp
#include "erom/extraction.hpp"

using namespace mhs::app::erom;
auto result = extract(definition, mesh, {{0, {5, 200}}, {1, {500, 500}}}, 1e-3);
write_erom("package.xml", result.asset);
```

Extraction supports nonempty solid-only models and diagonal anisotropic
conductivity. Material properties are sampled at their cell centers and
boundary fields at facet centers with `T = InitialTemperature` (the FOM
reference temperature), `t = 0`. Spatial variation is retained in the original
coordinates. Invalid or nonfinite values and nonpositive material properties
are rejected. A referenced fluid block is unsupported. Extraction adapts a temporary model
input and compiles it through the ordinary FOM compiler. Frozen prescribed
temperature fields and the constant conductivity stencils needed for probe
projection are prepared by the extractor; the compiler and Green–Gauss
recovery kernel require no extraction-specific mode.

Unselected convection, prescribed-temperature, prescribed-flux and default
boundary conditions are assembled using the frozen FOM discretization.
Their conductance is included in the projected K and their temperature-rise
load in the projected fixed load f. Each thermally disconnected region needs
at least one positive fixed or trained conductance. Neumann flux alone does
not anchor a region. At least one explicit trained group is still required.

Each explicit source with active cells uses a unit total-power spatial input:
the weight of each cell is its volume divided by the sum of active volumes
with that source ID. The source's volume-weighted mean temperature is also
retained. Zero-power declarations are not removed. All omitted instance
power schedules default to zero.

All requested point probes use the frozen FOM Green–Gauss and boundary trace
reconstruction. Their affine output offsets retain fixed boundary effects.
Probes outside the active model or depending directly on a replaceable
boundary trace are rejected with their names. After extraction, interface,
outline and probe coordinates are translated by the active model's AABB
minimum; no rotation, scaling or original origin is stored.

Extraction uses the Extended-FANTASTIC response-space algorithm. Physical
heat-source columns and a nonzero frozen fixed load drive training. A passive
region uses its physical ambient-temperature inputs on selected HTC surfaces
or its fixed conductance input if it has no trained surface. The actual
frozen default load also supplies response snapshots when nonzero. Global capacity-scaled spectral
estimates use a preconditioned low-end Rayleigh–Ritz iteration and restarted,
reorthogonalized high-end Krylov iteration. Both require a checked eigenpair
residual. Elliptic matching points are positive real shifts, evaluated with
arithmetic-geometric means using the complementary modulus. Training visits
those shifts from high to low, matching the independent Python reference.
Each input keeps its own response space. Ten successful log-uniform HTC
residual probes advance each input to the next shift. The fixed C++ random
seed gives reproducible native runs.

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
At each sampled setting it validates individual power, ambient and fixed-load
responses, exploiting symmetry to solve per output when there are fewer
outputs than independent inputs. The default case uses zero explicit power,
facet-level default HTC/ambient values and affine output offsets. The measured error is the
temperature difference divided by the larger of one kelvin and the magnitude
of the FOM temperature rise. Accuracy after embedding depends on the attached
geometry and loads.

## Native EROM XML v1

The root is `<Erom version="1">`, encoded as UTF-8. All geometry and thermal
quantities use SI units. Numeric arrays are whitespace-separated decimal
values, row-major for matrices, written at double max_digits10 precision
independently of locale. Matrix attributes are `rows` and `cols`; vector
attributes are `size`. Zero-column and zero-row matrices are valid.

| Required element | Contents |
| --- | --- |
| ReferenceTemperature | Reference temperature in K |
| Operators | K[n,n], C[n,n], B[n,ns], L[no,n], Vb[nb,n], FixedLoad[n], OutputOffset[no] |
| UniformTemperatureMode | Vector of length n |
| BoundaryGroups | BoundaryGroup records: id, name, htcMin, htcMax |
| Interfaces | Interface records: boundary, resistance, defaultHtc, ambientTemperature, facet attributes |
| Outline | Group records (name), each containing Facet records |
| Sources | Source records: id, name |
| Outputs | Output records: id, name, optional x/y/z coordinates (all three or none) |

Each facet has attributes `axis` (x/y/z), `normal` (-1/+1), `coordinate`,
`uMin`, `uMax`, `vMin`, `vMax`. Tangential axes are YZ, XZ and XY respectively.
Interface order is Vb row order. `boundary` refers to the zero-based trained
group ID. Source and output IDs are contiguous and follow matrix column/row
order. Outline facets only diagnose contact; they cannot be assigned new
environment laws. The facet-level default HTC and ambient temperature are
discrete frozen values retained on exposed patches. Instance overrides are
group-level; their HTC need not lie within the training interval.

The reduced state satisfies `C dq/dt + (K + K_external)q = BP + f + f_external`.
Interface cell-side temperature is `T_ref + Vb q`; output temperature is
`T_ref + L q + OutputOffset`. Fixed contributions remain in K and f regardless
of how the package is placed. The reader checks version, fields, counts,
finite data, dimensions, positive-definite C, positive-semidefinite K, facet
geometry and references. Legacy binary assets are unsupported and must be
re-extracted. The sampled validation tolerance is not a rigorous guarantee of
all transient trajectories or arbitrary embedded boundary conditions.

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

The EROM figures used a reduced-only contact cache. The current assembler
computes contact terms in the shared global path for all projects with EROMs
or multiple FOMs.
