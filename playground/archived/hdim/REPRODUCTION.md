# Reproduction of the HDI paper (ISEDA 2025)

Target: S. Lan, M. Tang, J. Mao, *Hierarchical Decomposition and Interconnection Method
for Efficient Thermal Simulation of Chiplet-Based 2.5D System*, ISEDA 2025,
DOI 10.1109/ISEDA65950.2025.11100833. Paper PDF and text extract are in this directory.

The model is Table I of the paper, digit for digit: twelve layers with the published
size, `k_parallel`/`k_perpendicular`, `cp` and `rho`; `kappa_die = 148 (T/300)^-1.3`;
ambient 300 K; `h = 200 W/(m^2 K)` on the heat sink top and `h = 10 W/(m^2 K)` on the PCB
bottom, everything else adiabatic; chip1 = Multi-GPU (125.8 W, Fig. 4), chip2 = AI
training (58.3 W, Fig. 5); `Delta t = 1 s`, 200 s, the two alternating cases of Sec. III-B.
What the paper does not publish is reconstructed here and marked as such: the chiplet
coordinates inside the 45 x 35 mm outline (Figs. 4-5 give sizes and power shares but no
origin), the gap between chiplets, the mesh rule (Fig. 9 gives unknowns, not a mesh), the
observation points P1-P3 (Fig. 8 gives no coordinates) and the interface partition P of
eq. (7)-(8).

## Status

| part | state |
| --- | --- |
| Table I stack, chiplet layouts, boundary conditions | implemented |
| standard FVM of Sec. III over the whole stack, fixed-point iteration on `kappa(T)` | implemented, converged, used as the reference |
| mesh levels reproducing the five DoF counts of Fig. 9 | calibrated, within 1% (table below) |
| steady state of Sec. III-A | reproduced, hotspot 407.6 K (> 400 K) |
| transient of Sec. III-B, both alternating cases | reproduced, peaks 350.3 K / 356.9 K |
| HDI: R of eq. (8) and T0 of eq. (7), zero-input response of eq. (4) | implemented |
| HDI: interconnection of eq. (10) inside the nonlinear solve | **not converged**, see below |
| Fig. 9 timings and the 48x speed-up | not produced: the baseline here is AMG-CG, and the HDI solver does not converge yet |

## Geometry and mesh

`mhs_hdim --level k` prints the assembled model. The in-plane size of each level was
calibrated by bisection (`--probe`) so that the cell count, which is the unknown count of
this discretisation, meets the DoF of Fig. 9:

| level | in-plane | die/thin cells | unknowns | Fig. 9 |
| --- | --- | --- | --- | --- |
| Mesh1 | 0.5983 mm | 1 / 1 | 64,408 | 65,760 |
| Mesh2 | 0.2839 mm | 2 / 1 | 308,484 | 304,640 |
| Mesh3 | 0.2154 mm | 2 / 2 | 914,448 | 912,640 |
| Mesh4 | 0.1560 mm | 3 / 2 | 1,819,088 | 1,815,040 |
| Mesh5 | 0.1397 mm | 4 / 3 | 3,315,648 | 3,296,640 |

Mesh1 holds 64,408 cells split into 13,224 / 18,088 / 33,096 (nonlinear / top linear /
bottom linear), 8,816 interface faces and 184.101 W of total power, which is the
125.8 W + 58.3 W of the paper.

## Sec. III-A: steady state

Both chips at full workload, `mhs_hdim --level 0 --steady`:

```text
[steady] ref   sweeps  3    0.52 s  hotspot 407.57 K in chip1 at (-8.6, 2.7) mm  chip1 407.57 K  chip2 406.54 K
```

* the hotspot is 407.6 K, i.e. above the 400 K the paper reports, and it sits in chip1
  inside the footprint that the AI core of chip2 covers from above — the mechanism the
  paper describes for the CPU hotspot;
* chip2 is cooler than chip1, as in Fig. 7;
* Mesh2 gives 407.60 K and 406.60 K, i.e. 0.03 K away from Mesh1: the steady answer is
  mesh converged, so the 407.6 K is a property of the model, not of the grid.

Nothing in Table I was tuned to reach 400 K. The resistance that produces it is the
spreading resistance of the 1 mm, 60 x 60 mm pedestal and of the 6.9 mm sink: the chips
inject 184 W over 45 x 35 mm and the convection surface is 100 x 100 mm, which gives
about 0.58 K/W, the same order as the 0.54 K/W implied by the paper's own "> 400 K".

## Sec. III-B: transient state

200 s of 1 s steps, `mhs_hdim --level 0 --transient`. Case a: chip1 at full workload for
the first 100 s and chip2 idle, then the reverse. Case b: the reverse order.

| case | paper | measured (Mesh1) |
| --- | --- | --- |
| a, chip1 first | peak at t = 100 s, below 350 K | 350.30 K at t = 100 s |
| b, chip2 first | close to 360 K at t = 200 s | 356.87 K at t = 200 s |
| peak deviation between the two orders | nearly 10 K | 6.6 K |

Observation points (the paper gives none; these are the chiplet centres of Figs. 4-5):
P1 = CPU of chip1 under the AI core, P2 = GPU0 of chip1, P3 = AI core of chip2.

| case | P1 | P2 | P3 | hotspot |
| --- | --- | --- | --- | --- |
| a | 348.73 K @100 s | 349.80 K @100 s | 348.32 K @100 s | 350.30 K @100 s |
| b | 355.30 K @200 s | 356.36 K @200 s | 354.89 K @200 s | 356.87 K @200 s |

The peak order follows the paper: heating the chip that runs first and keeping it idle
afterwards leaves the package colder than running the other chip first, because the
second half of case b starts from the package already warmed by 100 s of chip2.

## HDI method: what works and what does not

Implemented: the three-region split of Sec. II-A, the macromodel of eq. (7)-(8) — one
unit-power excitation per interface block, block-averaged response, symmetrised — the
zero-input response of eq. (4) and the zero-state completion of eq. (5), and the
interconnection attempt of eq. (10) as the Robin boundary condition of the nonlinear
region.

Not converged: the interface loop that couples the nonlinear region with the macromodel.
Three formulations were tried on Mesh1 with 616 blocks per interface, all measured
against the reference:

1. Robin condition with the datum `T_I` and the series conductance on the diagonal,
   `T_I` updated through the exact closure `q = (I + G R)^-1 G (t - c)`: stable but
   crawls, the change falls by a factor 0.985 per sweep (needs ~700 sweeps), and the flux
   stays at 0.2 W instead of 184 W.
2. Defect correction with the diagonal `1/(1/g + R_bb)`: diverges, a factor 3 per sweep.
3. Robin condition with the datum `T_I` and the flux `q = g (t - T_I)` of the last solve:
   also diverges, the temperature of the nonlinear region runs away.

Diagnosis: the interface operator `J = dq/dt = (I + G R)^-1 G` has a local scale set by
the block self-resistance (113 K/W at the bottom interface, 2.7 K/W at the top) and an
aggregate scale set by the row sum of R (308 K/W, i.e. 2 W/K for the whole interface).
No diagonal preconditioner covers both, so a staggered loop either over-damps the
aggregate mode or is unstable in the local modes; and because the nonlinear region has no
other boundary condition, a wrong interface conductance leaves its operator nearly
singular. The fix is to treat eq. (10) implicitly — the bordered system
`(K_N + S^T C^-1 A) t = b + S^T C^-1 G c` with `C = I + G R` — or to precondition the
staggered loop with the exact diagonal plus a rank-one aggregate term. With one block per
mesh face (`--interface-step 0`) the closure is local and the method collapses onto the
standard two-point flux, which is the exactness check to build first.

## Timings

Only the reference solver is timed here, because the HDI solver does not converge yet:

| level | unknowns | steady (3 sweeps) |
| --- | --- | --- |
| Mesh1 | 64,408 | 0.52 s |
| Mesh2 | 308,484 | 2.89 s |

The transient at Mesh1 takes 260 s for case a (200 steps, 505 linear solves) and 301 s for
case b (531 solves), i.e. about 1.4 s per step with an AMG-CG solve of the whole stack.
Fig. 9 of the paper reports 563 / 1050 / 2582 / 3479 / 5316 s for the same 200 s march and
an HDI speed-up of 6x / 16x / 23x / 30x / 48x; those numbers cannot be compared with
these, because the paper's in-house FVM baseline is not the same solver and its timings
were taken on two Xeon 8268.

## How to reproduce the numbers above

```shell
cmake -S playground -B build-playground -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=OFF
cmake --build build-playground --target mhs_hdim --parallel
cd build-playground/hdim
./mhs_hdim.exe --level 0 --steady
./mhs_hdim.exe --level 1 --steady
./mhs_hdim.exe --level 0 --transient          # ~10 min, both cases
./mhs_hdim.exe --level 0 --steady --hdi       # the HDI attempt, does not converge yet
```
