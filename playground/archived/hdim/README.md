# HDI reproduction (ISEDA 2025)

Reproduction of *Hierarchical Decomposition and Interconnection Method for Efficient
Thermal Simulation of Chiplet-Based 2.5D System* (S. Lan, M. Tang, J. Mao, ISEDA 2025,
DOI 10.1109/ISEDA65950.2025.11100833). The paper PDF and its text extract live in this
directory but are not committed; `REPRODUCTION.md` records what was reproduced and what
was not.

## What is here

| file | role |
| --- | --- |
| `stack.hpp` | Table I: twelve layers with size, `k_parallel`/`k_perpendicular`, `cp`, `rho`; the die conductivity `148 (T/300)^-1.3`; the ambient and the two convection coefficients; the chiplet layouts of Fig. 4 and Fig. 5; the five mesh levels of Fig. 9. |
| `mesh.hpp` | Rectilinear cell-centred grid: a graded in-plane axis shared by all layers, one cell count per layer through the thickness. |
| `model.hpp` | The three regions of Sec. II-A, per-cell power, the FVM assembly of eq. (2)-(3) split by region, the interface faces of eq. (7)-(10), and `assemble_whole`, the same discretisation as one global operator (the standard FVM of Sec. III). |
| `hdi.hpp`, `hdi.cpp` | The method of Sec. II: the macromodels (R of eq. (8), T0 of eq. (7)), the zero-input response of eq. (4), and the interconnection of eq. (10) as the Robin condition of the nonlinear region. |
| `reference.hpp`, `reference.cpp` | The standard FVM over the whole stack with the fixed-point iteration on `kappa(T)`: accuracy reference and timing baseline. |
| `main.cpp` | Driver: geometry, the steady state of Sec. III-A, the two alternating transient cases of Sec. III-B, and the comparison between the two solvers. |

## Build and run

The driver is part of the standalone playground build, which does not touch the main
`CMakeLists.txt`:

```shell
cmake -S playground -B build-playground -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=OFF
cmake --build build-playground --target mhs_hdim --parallel
cd build-playground/hdim
./mhs_hdim.exe --level 0 --steady        # Sec. III-A with the standard FVM
./mhs_hdim.exe --level 0 --transient     # Sec. III-B, both alternating cases
```

Options:

```text
--level N              mesh level of Fig. 9, 0 (Mesh1) to 4 (Mesh5)
--steady               steady state of eq. (2) only
--transient            the 200 s march of eq. (3) only
--steps N --dt X       time grid, default 200 steps of 1 s
--interface-step X     interface block size in mm for the macromodel, 0 = one block per face
--sweeps N             nonlinear sweep budget, default 50
--tolerance X          largest temperature change accepted between two sweeps, default 1e-4 K
--hdi                  also run the HDI method and compare it with the reference
--linear-tolerance X   tolerance handed to the linear solver
```

`--hdi` is experimental: the pre-simulation and the zero-input/zero-state split work,
the interface loop of eq. (10) does not converge reliably yet. See `REPRODUCTION.md`.

## Conventions

* The stack runs from the heat sink at the top (`kStack[0]`) down to the PCB
  (`kStack[11]`); the in-plane axes span `[-50, 50] mm` with the chip outline centred on
  the origin, and every chiplet coordinate is relative to that outline.
* `region_cells[r]` holds the global cell ids of a region, and every linear system is
  written in the ordering of that vector; the interface faces carry the cell ids, so a
  flux can be scattered back to either side.
* The interface partition (`--interface-step`) is this reproduction's own choice: the
  paper states that R is P x P over the face elements of the interface but never gives
  P, and a per-face R at Mesh5 would be a dense 81k x 81k matrix (52 GB). The default is
  a 1.6 mm block, i.e. 616 blocks per interface; `--interface-step 0` reproduces the
  standard FVM exactly, because then eq. (10) collapses to the two-point flux of the
  face.
