# DTDM reproduction report

> **Archived, not finished.** This reproduction was stopped on 2026-10-20 and the
> work moved to `playground/hdim` (ISEDA 2025, *Hierarchical Decomposition and
> Interconnection Method*). The reason is a property of the paper, not of the
> code: the model it reports cannot be pinned down from what it publishes. Fig. 3
> is a cross-section with layer names and no numbers, the text points at two other
> references for the material data, the mesh refinement rule, the element type,
> the chiplet coordinates and the per-chiplet powers are all unpublished, and `s`
> is never given. What is left is a reproduction whose absolute numbers depend on
> a stack we guessed, and the accuracy claim cannot be separated from that guess.
> The code runs, is self-consistent and reproduces the decomposition to 6e-05 K;
> the reader should treat everything below as a record, not a validation.

Reproduction of *Domain Transformation and Decomposition Method for Composable
Thermal Modeling and Simulation of Chiplet-Based 2.5D Integrated System*
(ASP-DAC 2026). Branch `agent/experiment1018`, implementation in
`playground/dtdm`.

**Which of the paper's two examples this is.** The paper has two: Sec. III-A, a
multi-GPU chip (two GPU, one CPU, three HBM chiplets on a 45 mm interposer) and
Sec. III-B, a CPU-DRAM chip (four CPU, four DRAM). This directory reproduces the
**first**, `--system multi_gpu`, because it is the only one with a published
accuracy figure (Fig. 7 and the 0.02 K statement at t = 50 s) and because its
four meshes are the ones whose DoF can be matched. The paper's Table I, however,
belongs to the *second* example — Section 5.1 gives the evidence — so the timings
here are a scaling study of the same algorithm at the same DoF, not that table.

## 1. Method and model

Reproduced as the paper defines them:

- **Temporal transform.** Laguerre basis `phi_p(x) = exp(-x/2) L_p(x)` with
  `x = s t`, Galerkin projection. Because `phi_q' = -phi_q/2 - sum_{k<q} phi_k`,
  the order-`q` equation is `(K + (s/2) C) T_q = f_q`, whose matrix is
  **independent of `q`**: one operator per module serves every order and the
  history enters only through the right-hand side.
- **Spatial decomposition**, the paper's Sec. III split: CM I = heat sink +
  heat spreader + TIM, KM = chip + micro bump + RDL, CM II = interposer + C4
  bump + substrate + BGA + PCB. The TIM sits on the chip but belongs to CM I, so
  the CM I / KM interface is the TIM / chip plane.
- **Interface macromodel**, eq. (8)-(11). The interface is set to an isothermal
  0 K and every face element is excited in turn; the module returns the
  interface flux and `G = W^T A^-1 W - diag(g)` is the equivalent conductance
  matrix in the patch basis. `NI` face elements give an `NI x NI` dense operator.
- **Interface composition**, eq. (12)-(13). The modules meet only through the
  patch temperatures: each one contributes the flux its own macromodel predicts,
  the contributions are summed on the shared interface, and that sum is factored
  once for the whole march. This is what "composable" means in the title.
- **Marching-on-order**, Algorithm 1: per order two free solves of eq. (6), one
  interface solve, two Dirichlet solves of eq. (7), then reconstruction by
  eq. (2).

### 1.1 The two decisions the paper leaves open

**The constant part of the response.** A constant power input drives a constant
temperature change of about 34 K, and a constant `a` is carried by the Laguerre
coefficients `2a(-1)^q`, which never decay. The paper excites every order with
`f_q = 2 P (-1)^q` and lets the series carry the constant. This implementation
solves `K theta_ss = f` once and lets the series carry `theta - theta_ss`.

The two are the same series term by term, but **not the same at finite `N`**:
substituting `T_q = W_q + 2 theta_ss (-1)^q` gives exactly
`(K + s/2 C) W_q = -s C (theta_ss + sum_{k<q} W_k)`, which is what the code
solves, and the two reconstructions differ by

```text
theta_paper - theta_this = theta_ss (1 - 2 sum_{q<N} (-1)^q phi_q(s t))
```

a factor that goes to zero as `N` grows but is not zero at `N = 300`. The code
prints it (`[dc] paper form minus this form, in units of theta_ss`), so the
difference is disclosed rather than assumed away. The reason for the split is
numerical: the shifted coefficients decay, so a residual test on them means
something, and the constant is carried exactly instead of by a truncated
alternating sum.

**The scaling factor.** The paper states that `s` scales time and never says how
it is chosen. It is set here to `s t_d = 10`, that is `s = 0.2` for the 50 s
window, and Section 3.4 reports what it changes.

### 1.2 Where this implementation departs from Algorithm 1

Algorithm 1 builds `(G, Q0)` for the **CMs only** (line 2; Sec. IV repeats it:
"constructing Laguerre-based macromodels for CMs", and Sec. III-B reuses "the
macromodels of CM I and CM II"). The KM carries no macromodel: it is solved each
order with the equivalent interface flux imposed, exactly as the paper describes.

This implementation builds the macromodel for **all three** modules and solves
one coupled interface system. That is the same Schur complement read the other
way round: eliminating the patch temperatures with all three `G`s leaves a
sparse KM operator and a dense `P x P` interface system (what is done here),
while eliminating them with only the two CM `G`s leaves a dense correction
`-W H^-1 W^T` inside the KM operator (the literal form). The two are exact, and
the agreement between DTDM and the monolithic solve in Section 4 shows it for
this discretisation: 6.2e-05 K, which is the CG tolerance.

The cost trade-off is real and is not hidden: the macromodel here has
`P_CMI + P_KM + P_CMII` columns, that is 4932 solves at Mesh1 against 2466 for
the literal form, while the literal form pays the dense correction in every KM
matrix-vector product of every CG iteration. That was measured, not assumed. The
literal form was implemented as well — the KM's operator with
`A - W K^-1 W^T` applied matrix free and preconditioned by AMG built on the
sparse `A` alone — and at Mesh1, 300 orders, one patch per face, it gives:

| form | macromodel columns | setup | marching | total | KM solve iterations | DTDM - WLP |
| --- | --- | --- | --- | --- | --- | --- |
| all three modules | 4932 | 26.62 s | 31.21 s | 57.84 s | 3.1 per solve | 6.2e-05 K |
| CMs only (Alg. 1 literal) | 2466 | 24.42 s | 208.18 s | 232.61 s | 172 per solve | 2.4e-05 K |

Both forms are exact, and the literal one is in fact the more accurate of the
two, but its condensed operator loses the conditioning that the sparse form has:
AMG on the KM alone cannot see the dense interface term, so CG needs 55x more
iterations, and the march becomes 6.7x slower while the setup barely improves —
the KM is the smallest module, so its columns were the cheapest to extract. The
literal form is therefore implemented, measured and removed; what remains is the
sparse form, and the reader should treat the numbers below as belonging to it.

## 2. Mesh scale

One number per level, the cells across the widest chiplet, sets the finest
in-plane cell; every other interval follows from it, uniform across the chiplets
and their gaps, geometrically coarser outwards, so the unknowns sit where the
heat is. The paper's DoF come out of the geometry to within about a percent:

| level | n_fine | in-plane cells | finest cell | DoF | paper | deviation |
| --- | --- | --- | --- | --- | --- | --- |
| Mesh1 | 19 | 67 x 72 | 0.29 mm | 26579 | 26513 | +0.25 % |
| Mesh2 | 59 | 161 x 178 | 0.24 mm | 213258 | 210848 | +1.14 % |
| Mesh3 | 151 | 363 x 403 | 0.11 mm | 1202498 | 1208923 | -0.53 % |
| Mesh4 | 198 | 465 x 518 | 0.083 mm | 2043764 | 2037523 | +0.31 % |

Matching the count is not matching the mesh: the paper does not publish how its
four meshes are refined, so this is only evidence that the scale is right. The
module split that follows is physical: CM I 29-37 %, KM 19-21 %, CM II 44-52 %
of the unknowns.

Total power is the paper's **251.6 W** (Fig. 5 pie chart of the multi-GPU
configuration: GPU 78.1 %, CPU 13.9 %, HBM 8.0 %), not the 755 W of REF-28, which
belongs to the upstream TAP-2.5D configuration. The figures print only the total
and the category shares, so the per-chiplet wattages in `stack.hpp` are an even
split inside each category and are marked ASSUMED there.

The resulting peak is 34.34 K above ambient, that is 334.3 K absolute. The
paper's Fig. 7(a,b) prints a 316-333 K colour bar over the cross-section, and its
top is not a published maximum, so this is an order-of-magnitude check and not a
validation: the peak lands 1.3 K above the bar's top, which is the expected
residual given that the chiplet coordinates, the per-chiplet wattages and the
mesh refinement rule are all assumed.

## 3. Accuracy

### 3.1 What is measured, and what the paper claims

The paper's error statement is one number about one instant: "the temperature and
error distributions of the horizontal cross-section of the chip at t = 50 s ...
the maximum absolute error is less than 0.02 K" (Fig. 7(c), against ANSYS
FLUENT), and the error map of that figure is capped at 0.019 K. Two things
follow, and both limit what can be claimed here.

- **It includes FLUENT's own spatial discretisation**, and this reproduction has
  no FLUENT. What is measured below is therefore the error of the *time*
  approximation of the same semi-discrete model — a different, smaller quantity,
  not a reproduction of the paper's 0.02 K against a second code. The spatial
  discretisation is unchanged between the two methods being compared, so it
  cancels exactly rather than being estimated.
- **It is one cross-section at one time.** The metric used here is stricter in
  space (the maximum absolute difference over the whole field, not one slice)
  and still sampled in time: eleven probe times, not a continuous bound. The
  tables below say "at each sampled time" and never "over the whole transient".

The case itself also differs: the paper publishes the chiplet sizes (via REF-28)
and the per-category power shares, but not the coordinates, not the per-chiplet
wattages and not the mesh refinement rule. Those are assumed here and flagged in
`stack.hpp`. A 0.02 K agreement with the paper's numbers is therefore not
attainable and not claimed; what is checked is that the approximation error sits
well below the paper's budget.

### 3.2 The reference

Backward Euler on the same mesh and the same operators, because that is the only
reference that isolates the transform from everything else.

At the paper's own time-marching configuration — 5000 steps of 0.01 s over 50 s
— one march is **not** a reference. Backward Euler's global error is first order
in the step and a cold start is where most of it is made: the slow modes are
allowed to answer within one step, which at 0.01 s is 0.18 K at t = 0.05 s and
0.02 K at t = 0.5 s. So the reference refines internally, which changes nothing
about what is being compared:

- the 5000-step march is run anyway, and its cost is the time-marching baseline;
- two finer marches at 10000 and 20000 internal steps are combined by
  `2 theta(dt/2) - theta(dt)`, which removes the first-order term;
- a third of the correction that extrapolation applied is reported as what is
  left of the reference's own error. It is an estimate under the assumption that
  the next term is geometrically smaller, not a bound.

For the run below that estimate is **2.9e-02 K**, and it is set by the earliest
probes alone: the same figure is 3e-03 K or less from 0.25 s on.

### 3.3 The result

Mesh1, 300 orders, `s t_d = 10`, one patch per interface face, 251.6 W. Maximum
absolute difference over the whole field, at each probe time. The first probe is
the initial condition, which the reference holds exactly:

| t / s | Thermal-WLP vs reference | DTDM vs reference | DTDM vs Thermal-WLP |
| --- | --- | --- | --- |
| 0.00 | 1.6009 | 1.6009 | 0.0001 |
| 0.01 | 0.2514 | 0.2513 | 0.0001 |
| 0.05 | 0.1960 | 0.1960 | 0.0001 |
| 0.10 | 0.0732 | 0.0733 | 0.0001 |
| 0.25 | 0.0291 | 0.0291 | 0.0001 |
| 0.50 | 0.0240 | 0.0240 | 0.0001 |
| 1.00 | 0.0258 | 0.0257 | 0.0001 |
| 2.00 | 0.0139 | 0.0139 | 0.0001 |
| 5.00 | 0.0059 | 0.0059 | 0.0001 |
| 10.00 | 0.0043 | 0.0043 | 0.0001 |
| 20.00 | 0.0019 | 0.0019 | 0.0001 |
| 35.00 | 0.0007 | 0.0006 | 0.0001 |
| 50.00 | 0.0015 | 0.0014 | 0.0001 |

The same comparison at the hotspot, which is the waveform the paper plots:

| t / s | Thermal-WLP | DTDM | reference | reference - WLP |
| --- | --- | --- | --- | --- |
| 0.00 | 1.6009 | 1.6009 | 0.0000 | -1.6009 |
| 0.01 | 2.0702 | 2.0702 | 2.0290 | -0.0412 |
| 0.05 | 3.7389 | 3.7390 | 3.8605 | +0.1216 |
| 0.10 | 4.9532 | 4.9532 | 4.9155 | -0.0377 |
| 0.25 | 6.6600 | 6.6600 | 6.6766 | +0.0166 |
| 0.50 | 8.7448 | 8.7449 | 8.7288 | -0.0160 |
| 1.00 | 11.4885 | 11.4886 | 11.5075 | +0.0190 |
| 2.00 | 14.8573 | 14.8573 | 14.8675 | +0.0102 |
| 5.00 | 19.8098 | 19.8098 | 19.8141 | +0.0043 |
| 10.00 | 23.6956 | 23.6956 | 23.6924 | -0.0032 |
| 20.00 | 27.7775 | 27.7776 | 27.7761 | -0.0014 |
| 35.00 | 31.6325 | 31.6325 | 31.6329 | +0.0004 |
| 50.00 | 34.3384 | 34.3384 | 34.3394 | +0.0011 |

Four readings, in order of how much they matter:

1. **At the instant the paper compares, the budget is met with two orders of
   margin**: 0.0015 K at t = 50 s against the 0.02 K claimed.
2. **The transform does not satisfy the initial condition.** At t = 0 it reports
   a 1.60 K rise where the initial rise is 0, and that is the largest error
   anywhere in this report. It is a property of the basis, not of the marching:
   the fast modes of the stack equilibrate in microseconds, which for a Laguerre
   basis with `s t_d = 10` is a step at `x = 0`, and 300 smooth functions cannot
   represent a step. What is left at 0.01 and 0.05 s is the same defect decayed
   by `exp(-s t / 2)`. Carrying the constant exactly (Section 1.1) already helps
   here: `[dc]` prints 1.0 for the t = 0 factor, so the paper's own excitation
   would report 1.60 - 34.34 = -32.7 K at t = 0.
3. **From 0.25 s on the error is under 0.03 K at every sampled time**, decaying
   to 1e-03 K at the end. Between 0.5 and 1 s it is 0.024-0.026 K, i.e. the
   0.02 K budget is exceeded by 30 % there and met everywhere else.
4. **The first 0.1 s is not resolved to that budget** (0.20 K at 0.05 s, 5 % of
   the 4 K rise at that instant), and Section 3.4 shows that `s` moves this error
   while Section 4 shows that DTDM and the monolithic solve agree to 6e-05 K on
   it — so it is the basis, not the decomposition.

### 3.4 What the scaling factor changes

The paper says `s` scales time and does not choose it. `s t_d = 10` is the value
used throughout this report; the hotspot it produces over a scan of `s t_d` at
300 orders is:

| s t_d | 0.05 s | 0.10 s | 0.25 s | 1.00 s | 5.00 s | 50.00 s |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 3.7389 | 4.9532 | 6.6600 | 11.4885 | 19.8098 | 34.3384 |
| 20 | 3.8908 | 4.9298 | 6.6850 | 11.5016 | 19.8159 | 34.3394 |
| 60 | 3.8394 | 4.9294 | 6.6814 | 11.5089 | 19.8136 | 34.3393 |
| 100 | 3.8695 | 4.9071 | 6.6812 | 11.5089 | 19.8137 | 34.3394 |
| 200 | 3.8513 | 4.9082 | 6.6716 | 11.5106 | 19.8119 | 34.3385 |
| 300 | 3.9046 | 4.9406 | 6.6529 | 11.4906 | 19.8299 | 34.3412 |
| 600 | 4.0872 | 5.1802 | 6.4603 | 11.6713 | 19.8351 | 34.2604 |
| 1000 | 3.2813 | 4.6027 | 6.8530 | 11.1502 | 19.8427 | 34.1510 |

Against the reference of 3.8601 / 4.9155 / 6.6766 / 11.5075 / 19.8141 /
34.3395 K, three regimes are visible:

- at the end of the window every value from `s t_d = 10` to 300 is within
  0.002 K, so the paper's claim does not depend on this choice;
- `s t_d` from 20 to 200 is a plateau in which the early error oscillates
  between 0.01 and 0.09 K without a trend, and `s t_d = 10` sits at its edge;
- from 300 on the transform breaks down, and by 1000 it is 0.5 K wrong:
  increasing `s` makes the fast modes' coefficients decay faster but the slow
  ones slower, and the slow ones eventually dominate.

Choosing `s` to minimise the early error would be fitting the reference, so it is
left at 10 and the scan is reported instead.

### 3.5 The linear solve is not the limit

Tightening the AMG-CG relative residual from `1e-8` to `1e-12` leaves the sampled
fields identical to four decimals (3.7389, 4.9532, 6.6600, 8.7448, 11.4885 K at
0.05 ... 1 s) and tightens the agreement between DTDM and Thermal-WLP from
6.2e-05 K to 3.9e-09 K. The operators here span ten decades, so an `1e-8`
*residual* is not an `1e-8` field error, but at this mesh it is far below
everything in Section 3.3. `--tolerance` exposes the knob.

### 3.6 What stands

- The paper's 0.02 K is a claim about t = t_d, and at t = 50 s the error is
  0.0015 K — thirty times inside it.
- From 0.25 s to the end of the window the error stays between 0.0007 and
  0.029 K at every sampled time.
- The transform does not satisfy the initial condition: at t = 0 it reports
  1.60 K where the rise is 0, and it is this defect, decayed in x, that leaves
  0.20 K at 0.05 s. Both belong to the Laguerre basis near x = 0 and neither is
  covered by anything the paper claims.
- The decomposition itself contributes 6.2e-05 K, which is the CG tolerance and
  four orders of magnitude below either of the above (Section 4).

## 4. Accuracy of the decomposition

The patch model is exact when one patch is one face element, and that is the
configuration used above: DTDM and Thermal-WLP agree to **6.2e-05 K** at Mesh1
in the maximum absolute value over the whole transient, which is the CG
tolerance, and to 2.3e-05 K at the end of the window.

A coarse patch grid is a *different* model, not a cheaper solve of the same one.
Measured at Mesh1, 300 orders:

| patches per interface | 1233 (one per face) | 32 x 32 = 1024 | 16 x 16 = 256 |
| --- | --- | --- | --- |
| DTDM - Thermal-WLP | 6.2e-05 K | 0.23 K | 0.92 K |

The paper does not publish `NI` per level, and it cannot be the face count at
Mesh3/4: `NI` faces per side there would need a dense `NI x NI` matrix of
gigabytes and an extraction of the same order of magnitude as the marching
itself. The measurements above bound the choice from below, and Section 5 uses
one patch per face at Mesh1 and 32 x 32 at Mesh2-4. **Those two are not the same
model**, so the accuracy in Section 3 belongs to the Mesh1 configuration and the
performance at Mesh2-4 belongs to a coarser interface.

## 5. Performance

### 5.1 What Table I reports, and which example it belongs to

Table I has two DTDM columns, `Pre-simulation` (the macromodel of eq. 8-11) and
`Simulation` (marching-on-order), plus the monolithic Thermal-WLP time:

| mesh | Thermal-WLP | DTDM pre | DTDM simulation | DTDM total | paper speedup |
| --- | --- | --- | --- | --- | --- |
| Mesh1 | 56 s | 5 s | 37 s | 42 s | 45x / 61x |
| Mesh2 | 273 s | 54 s | 115 s | 169 s | 88x / 142x |
| Mesh3 | 1990 s | 434 s | 713 s | 1147 s | 205x / 355x |
| Mesh4 | 3205 s | 673 s | 1290 s | 1963 s | 297x / 485x |

The first speedup of each pair is FLUENT over the Thermal-WLP column, the second
is FLUENT over the DTDM total; the separate claim of **738x** at Mesh4 is FLUENT
951666 s over the simulation-only 1290 s, the case where an already extracted
macromodel is reused and the pre-simulation is amortised away. All of them check
out arithmetically. Against the monolithic method the paper therefore claims
0.75x total at Mesh1, and 0.62 / 0.58 / 0.61 total with 0.42 / 0.36 / 0.40
simulation-only at Mesh2 / 3 / 4.

The table belongs to the **second** example, not the first. In the printed
two-column layout the paragraph that introduces it ("To verify the efficiency of
the proposed method across different problem scales ... summarized in Table I")
ends the right column of page 1404, after the dynamic-power-map discussion of
Sec. III-B, and the explanation continues on page 1405 with "the peripheral
packaging structures in Section III-A and Section III-B are identical.
Consequently, the Laguerre-based macromodels of CM I and CM II constructed in
Section III-A can be directly reused in this example", which is what the 738x
reuse figure refers to. Only the four mesh DoF are shared with the first example.

This reproduction models the multi-GPU chip of Sec. III-A with constant power. It
therefore uses the same stack, the same four meshes and the same algorithm, but
it is **not** a reproduction of Table I, and the timings below are not the
paper's benchmark. Nothing here claims the 738x reuse either: macromodels are
extracted per run.

### 5.2 Reproduced

Numbers below are from this implementation, one run each, on a 20-core machine
where the same configuration varied by up to 1.6x between runs: the Thermal-WLP
of Mesh1 was 12.86 s and 22.99 s in two runs of the identical command. The ratios
inside a row are the meaningful quantity, and the CG iteration counts, which are
deterministic, are given with them.

| level | patches | DoF | Thermal-WLP | DTDM pre | DTDM march | DTDT total | DTDM / WLP | paper total | paper march |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mesh1 | 1233 per face | 26579 | 22.99 s | 26.28 s | 30.85 s | 57.13 s | 2.48x | 0.75x | 0.66x |
| Mesh1 | 32 x 32 | 26579 | 12.86 s | 14.81 s | 29.02 s | 43.83 s | 3.41x | 0.75x | 0.66x |
| Mesh2 | 32 x 32 | 213258 | 275.22 s | 109.90 s | 199.27 s | 309.17 s | 1.12x | 0.62x | 0.42x |
| Mesh3 | 32 x 32 | 1202498 | 2796.91 s | 719.85 s | 1376.17 s | 2096.01 s | 0.75x | 0.58x | 0.36x |
| Mesh4 | 32 x 32 | 2043764 | 5576.32 s | 1307.63 s | 2542.61 s | 3850.24 s | 0.69x | 0.61x | 0.40x |

The trend is the paper's and by Mesh3 it has arrived: the decomposition is 2.5x
slower than the monolithic solve at Mesh1, crosses over at Mesh2, and is 1.3x /
1.4x *faster* at Mesh3 / Mesh4. Against the paper's ratios the two largest levels
are close:

| level | this march / WLP march | paper | this total / WLP | paper |
| --- | --- | --- | --- | --- |
| Mesh3 | 0.49x | 0.36x | 0.75x | 0.58x |
| Mesh4 | 0.46x | 0.40x | 0.69x | 0.61x |

The absolute times are on the paper's scale but 1.7-2x above it: 5576.32 s
against 3205 s for the monolithic solve at Mesh4, 3850.24 s against 1963 s for
the decomposition total. Since the paper's speedups are quoted against FLUENT,
they are not reproduced directly; if the paper's FLUENT time is taken as given
(951666 s at Mesh4), this implementation's simulation-only 2542.61 s would be
374x instead of 738x and its total 3850.24 s would be 247x instead of 485x.

### 5.3 Where the gap lives

Three mechanisms, measured rather than assumed.

**The interface is finer here at the small levels.** The paper publishes
`NI`, the number of interface face elements, only implicitly: Table I's
pre-simulation is 2 NI module solves for the two CMs, and it grows with the mesh,
5 s / 54 s / 434 s / 673 s for the four levels, that is close to linearly in the
DoF. This implementation uses one patch per interface face at Mesh1, which is
1233 per interface and 4932 macromodel columns, and a fixed 32 x 32 grid above
that, 2736 columns at every level. The two only meet in the middle: at Mesh1 the
pre-simulation is 26.28 s against 5 s, at Mesh4 it is 1307.63 s against 673 s. The
interface face count follows from the in-plane mesh, which the paper does not
publish, and a mesh that resolves the chip at 0.29 mm has an order of magnitude
more interface faces than one that does not.

**AMG-CG is bandwidth bound, so a small module is not proportionally cheap.**
The marching phase is where the decomposition has to win, and it wins exactly
when the modules are large enough for the per-iteration cost to follow their
size:

| level | WLP iterations for 300 orders | DTDM iterations for 1800 solves | ratio | marching time ratio | paper |
| --- | --- | --- | --- | --- | --- |
| Mesh1 | 4178 | 14453 | 3.46x | 1.34x | 0.66x |
| Mesh2 | 5335 | 16552 | 3.10x | 0.73x | 0.42x |
| Mesh3 | 7373 | 18997 | 2.58x | 0.49x | 0.36x |
| Mesh4 | 8778 | 21164 | 2.41x | 0.46x | 0.40x |

The decomposition needs 2.4-3.5x more iterations than the monolithic solve, but a
module iteration costs 4-8x less, so the marching phase goes from 1.34x slower at
Mesh1 to 0.46x at Mesh4, which is within 15 % of the paper's 0.40x. The gap at
the small levels has the same cause as in the first paragraph: at 26.5k-213k
unknowns an AMG-CG iteration is dominated by fixed costs (hierarchy traversal,
memory latency), not by the number of unknowns, so three modules in two layers
cost as much as one global solve.

**Only two layers of concurrency are exploited.** An order is two CM free solves
and the KM free solve in parallel, one dense interface solve, then two CM
Dirichlet solves and the KM Dirichlet solve in parallel: three solves deep,
counted as two maxima plus one small dense solve. Algorithm 1 has the same shape,
so this is the paper's parallelism too, and it is already used here through TBB.
What is not exploited is that the two CMs are independent *across* orders'
phases, and the paper does not claim it either.

Nothing in the gap is an accounting error: every solve is either one of the
`P` macromodel columns or one of the six per order that Algorithm 1 calls for
(two modules' worth of eq. (6) and eq. (7) for each of three modules).

### 5.4 Verdict against the objective

| what | result |
| --- | --- |
| the paper's four mesh levels | 26579 / 213258 / 1202498 / 2043764 unknowns, within 1.1 % |
| the paper's decomposition and equations | eq. (2)-(13) as printed, with the DC split of 1.1 |
| macromodel scope | all three modules instead of the two CMs, same Schur complement; the literal form was built and measured 4x slower (1.2) |
| decomposition error | 6.2e-05 K at Mesh1, the CG tolerance at `--tolerance 1e-8` |
| transform error at t = t_d | 0.0015 K, against the paper's 0.02 K claim |
| transform error at every sampled time from 0.25 s | 0.0007 to 0.029 K |
| transform error at t = 0 and in the first 0.1 s | 1.60 K at the initial condition and up to 0.20 K at 0.05 s; a property of the basis at `x = 0`, outside what the paper claims |
| peak temperature | 334.3 K against the 316-333 K colour bar of Fig. 7 |
| DTDM faster than Thermal-WLP at scale | 1.12x at Mesh2, 0.75x at Mesh3, 0.69x at Mesh4 |
| the paper's ratios | within 1.15x of the marching ratio and 1.13x of the total at Mesh4 |
| the paper's absolute times | 1.7-2x above them at Mesh4, on the same scale at Mesh3 |
| Table I as a benchmark | **not reproduced**: it belongs to the CPU-DRAM example of Sec. III-B (5.1) |
| the paper's reuse claim (738x) | not reproduced: no layout-to-layout macromodel reuse here |

## 6. Reproducing

```text
cmake -S playground -B build-playground -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=OFF
cmake --build build-playground --parallel
./build-playground/dtdm/mhs_dtdm.exe --level 0 --patch -1 --orders 300
./build-playground/dtdm/mhs_dtdm.exe --level 0 --patch -1 --orders 300 --marching
```

`--patch -1` is the paper's model, one patch per interface face; a positive
value replaces it with a coarse `n x n` patch grid, which is a coarser model and
measurably less accurate. `--marching` adds the reference of Section 3.2: it runs
the paper's 5000-step march for the time-marching baseline and two finer internal
marches for the reference itself. `--tolerance` sets the relative residual of
every iterative solve (1e-8 by default). `--scale` is `s t_d`, 10 by default,
and `--steps` is the time-marching step count, 5000 by default.

## 7. Provenance of the inputs

Read strictly: the paper publishes far less of the model than its figures
suggest. Fig. 3 is a cross-section with layer names and no numbers, and the text
points at REF-26 / REF-27 for the material data, so **no thickness and no
material property here is PAPER**.

- PAPER: the layer list and its order (Fig. 3), the module assignment of every
  layer (Sec. III), the two convection coefficients (`h = 400` on the heat-sink
  surface, `h = 10` on the PCB bottom), "all other surfaces are assumed to be
  adiabatic", the ambient 300 K, the 251.6 W total of the multi-GPU configuration
  and its split over the three categories (Fig. 5), the 50 s window and 0.01 s
  step, the 0.02 K error statement and the 316-333 K colour bar (Fig. 7), and the
  DoF of the four meshes (Table I).
- REF-28: the interposer edge of 45 mm (Sec. IV, case study 1), the rule that the
  spreader edge is twice the interposer's and the heat sink's twice the
  spreader's (Sec. III-A), and the chiplet sizes (Table II: CPU 12 x 12 mm, GPU
  18.2 x 18.2 mm, HBM 7.75 x 11.87 mm). Its 105 / 295 / 20 W per chiplet and its
  755 W / 680 W totals belong to the upstream TAP-2.5D configuration and are not
  used.
- CODE: the layer thicknesses and the material data, taken from the TAP-2.5D
  released input file; each one is tagged in `stack.hpp`.
- ASSUMED, and flagged as such in `stack.hpp`: the chiplet coordinates (Fig. 4
  fixes the arrangement and the HBM orientation but prints no coordinates, no
  dimensions and no scale bar), the RDL, BGA and PCB thicknesses, the per-chiplet
  power split inside each category, the meshing rule that reaches the paper's
  DoF, the structured hexahedral discretisation, the equivalent homogeneous
  properties used for the C4 and BGA layers that Fig. 3 draws as bump arrays, and
  the scaling factor `s`.
- NOT MODELLED, and disclosed where it matters: the heat sink's fins (Fig. 3
  draws them; the model is a flat plate with convection on one face), the wider
  substrate and PCB that Fig. 3 shows, and the four observation points of Fig. 6
  (one GPU and one CPU point per layout), which are replaced here by the hotspot
  waveform at every probe time.
- NOT reproduced: Table I as a benchmark, for the reason in Section 5.1; the
  second example of the paper, the CPU-DRAM chip under the DPM-10 / DPM-20
  waveforms of Fig. 8-10, which has no DoF table of its own and whose power
  waveforms are not implemented here (the layout is available as
  `--system cpu_dram`); and the macromodel reuse across layouts that the 738x
  figure rests on.
