# DTDM reproduction

Reproduction of *Domain Transformation and Decomposition Method for Composable
Thermal Modeling and Simulation of Chiplet-Based 2.5D Integrated System*
(ASP-DAC 2026): temporal WLP transform, spatial decomposition into three
composable modules coupled through interface macromodels, and
marching-on-in-order in the Laguerre domain.

The geometry, stack and mesh targets are taken from the paper. The four mesh
levels land within about a percent of the reported DoF and carry the paper's
251.6 W, the total of the multi-GPU configuration in Fig. 5.

This is the paper's **first** example (`--system multi_gpu`), the only one with a
published error figure. The paper's Table I belongs to the second, CPU-DRAM
example; `REPRODUCTION.md` Section 5.1 says why and what follows from it.

## Files

```text
stack.hpp   materials, 11-layer stack, chiplet layouts, the four mesh levels
mesh.hpp    axis generation and integer DoF search per mesh level
mesh.cpp    builds the mhs::core::Model and the mhs::sim operators
wlp.hpp     Laguerre basis and its derivative identity
dtdm.hpp    modules, interface patches, the three solution methods
dtdm.cpp    decomposition, macromodel extraction, marching-on-in-order
main.cpp    command line driver
```

## Build

The library is embedded with `add_subdirectory`, so nothing in the main tree
knows about this directory and the main build stays unaffected.

```text
cmake -S playground -B build-playground -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=OFF
cmake --build build-playground --target mhs_dtdm --parallel
```

## Run

```text
./build-playground/dtdm/mhs_dtdm.exe --level 0 --patch -1 --orders 300 --scale 10
```

| option | meaning |
| --- | --- |
| `--level` | 0..3 selects Mesh1..Mesh4 |
| `--patch` | patches per interface side; `-1` puts one patch on every interface face |
| `--orders` | Laguerre orders to march |
| `--scale` | the transform parameter, given as `s * duration` |
| `--duration` | time window in seconds, 50 s by default |
| `--solver` | `amg` (default) or `pardiso` when MKL is enabled |
| `--marching` | also run the time-stepping reference |
| `--steps` | time-marching steps, 5000 by default, i.e. the paper's 0.01 s over 50 s; the reference refines internally beyond it |
| `--tolerance` | relative residual of every iterative solve, 1e-8 by default |
| `--dof-only` | build the mesh only, to check the DoF against the paper |

## Mesh

One number per level, the cells across the widest chiplet, sets the finest
in-plane cell; every other interval follows from it, uniform across the chiplets
and their gaps and geometrically coarser outwards, so the unknowns sit where the
heat is:

| level | n_fine | in-plane cells | finest cell | DoF | paper | deviation |
| --- | --- | --- | --- | --- | --- | --- |
| Mesh1 | 19 | 67 x 72 | 0.29 mm | 26579 | 26513 | +0.25 % |
| Mesh2 | 59 | 161 x 178 | 0.24 mm | 213258 | 210848 | +1.14 % |
| Mesh3 | 151 | 363 x 403 | 0.11 mm | 1202498 | 1208923 | -0.53 % |
| Mesh4 | 198 | 465 x 518 | 0.083 mm | 2043764 | 2037523 | +0.31 % |

The module split that follows is physical: CM I 29-37 %, KM 19-21 %, CM II
44-52 % of the unknowns. An earlier revision of this directory reached the same
totals by giving the interposer margin 84 cells per side and the chip interior
five; that put almost every unknown in the periphery and left the KM at 0.2 %,
which is not a refinement of the model.

## Solvers

Every linear solve is iterative: AMG-preconditioned CG at `1e-8` relative
residual, warm started from the previous order of the same solve family. Each
module owns one solver on `K + (s/2) C` with its interface diagonal corrected
from the series to the half conductance.

The macromodel of eq. (11) is extracted by one solve per interface patch, and
only the `P x P` interface flux is kept, so the macromodel is `O(P^2)` in memory
and never `O(n P)`: `A^-1 W` is not stored. An order then costs two solves per
module, the free solve of eq. (6) and the Dirichlet solve of eq. (7), and the
modules run concurrently. The dense `P x P` system of eq. (12)-(13) does not
change with the order, so it is factored once.

Accuracy and timings are in `REPRODUCTION.md`.
