#!/usr/bin/env python3
"""Entry point for the transient BCI-ROM benchmark.

This module keeps the baseline research driver in ``_macromodel_demo_impl`` and
adds two repository-specific corrections:

* MetaHotspot authoring layers are registered top-first so the compiled mesh is
  physically ordered from substrate to cold plate.
* A source-aware, boundary-condition-independent residual Krylov basis is added
  to the four baseline reduction methods.  Its extraction uses only the
  isolated macro operators, a unit top-boundary operator, source footprint
  geometry, and Laplace shifts; no convection coefficient or ambient
  temperature is embedded in the basis.
"""

from __future__ import annotations

import _macromodel_demo_impl as impl

# Public aliases used by type annotations and the layer-building entry point.
PackageConfig = impl.PackageConfig
ExperimentConfig = impl.ExperimentConfig
MethodBasis = impl.MethodBasis
metahotspot = impl.metahotspot
GeometryOp = impl.GeometryOp
LengthUnit = impl.LengthUnit
Study = impl.Study
np = impl.np
sp = impl.sp
spla = impl.spla
scipy = impl.scipy
time = impl.time
asdict = impl.asdict
_axis_vertices = impl._axis_vertices
_layered_z_vertices = impl._layered_z_vertices
_add_materials = impl._add_materials
_add_full_rect = impl._add_full_rect
_chiplet_heat_source = impl._chiplet_heat_source


# ---------------------------------------------------------------------------
# Correct top-first authoring-layer registration
# ---------------------------------------------------------------------------


def build_package_model(
    cfg: PackageConfig,
    *,
    include_macro: bool,
    study: Study = Study.STEADY,
    duration_s: float = 0.0,
    output_interval_s: float = 0.0,
):
    """Build the representative package through the public scripting API."""
    model = metahotspot.Model()
    model.set_settings(
        study=study,
        length_unit=LengthUnit.MILLIMETER,
        initial_temperature_K=cfg.ambient_K,
        duration=duration_s,
        output_interval=output_interval_s,
    )
    model.set_mesh(
        _axis_vertices(cfg.width_mm, cfg.nx),
        _axis_vertices(cfg.height_mm, cfg.ny),
        _layered_z_vertices(cfg, include_macro),
    )
    _add_materials(model)

    # The compiler places the first authoring layer at the highest z interval.
    # Register the package from top to bottom so the physical mesh remains
    # substrate -> bumps -> die -> TIM -> spreader -> cold plate.
    if include_macro:
        cold_plate_layer = model.add_layer(f"{cfg.cold_plate_mm:.17g}")
        cold_plate = model.add_block(cold_plate_layer, "aluminum")
        _add_full_rect(model, cold_plate, cfg)

        spreader_layer = model.add_layer(f"{cfg.spreader_mm:.17g}")
        spreader = model.add_block(spreader_layer, "copper")
        _add_full_rect(model, spreader, cfg)

        tim_layer = model.add_layer(f"{cfg.tim_mm:.17g}")
        tim = model.add_block(tim_layer, "tim")
        _add_full_rect(model, tim, cfg)

    die_layer = model.add_layer(f"{cfg.die_mm:.17g}")
    mold = model.add_block(die_layer, "mold")
    _add_full_rect(model, mold, cfg)
    q_chiplet = _chiplet_heat_source(cfg)
    margin_x = 5.0
    margin_y = 5.0
    positions = (
        (margin_x, margin_y),
        (cfg.width_mm - margin_x - cfg.chiplet_width_mm, margin_y),
        (margin_x, cfg.height_mm - margin_y - cfg.chiplet_height_mm),
        (
            cfg.width_mm - margin_x - cfg.chiplet_width_mm,
            cfg.height_mm - margin_y - cfg.chiplet_height_mm,
        ),
    )
    for x, y in positions:
        chiplet = model.add_block(die_layer, "silicon", heat_source=q_chiplet)
        model.add_rect(
            chiplet,
            GeometryOp.ADD,
            f"{x:.17g}",
            f"{y:.17g}",
            f"{cfg.chiplet_width_mm:.17g}",
            f"{cfg.chiplet_height_mm:.17g}",
        )

    bump_layer = model.add_layer(f"{cfg.bump_mm:.17g}")
    underfill = model.add_block(bump_layer, "underfill")
    _add_full_rect(model, underfill, cfg)
    pitch_x = cfg.width_mm / cfg.bump_columns
    pitch_y = cfg.height_mm / cfg.bump_rows
    for iy in range(cfg.bump_rows):
        for ix in range(cfg.bump_columns):
            x = (ix + 0.5) * pitch_x - 0.5 * cfg.bump_width_mm
            y = (iy + 0.5) * pitch_y - 0.5 * cfg.bump_width_mm
            bump = model.add_block(bump_layer, "copper")
            model.add_rect(
                bump,
                GeometryOp.ADD,
                f"{x:.17g}",
                f"{y:.17g}",
                f"{cfg.bump_width_mm:.17g}",
                f"{cfg.bump_width_mm:.17g}",
            )

    substrate_layer = model.add_layer(f"{cfg.substrate_mm:.17g}")
    substrate = model.add_block(substrate_layer, "organic")
    _add_full_rect(model, substrate, cfg)

    # All external boundary operators are projected after extraction.
    model.set_default_neumann("0")
    return model


# ---------------------------------------------------------------------------
# Source-aware BCI port and residual spaces
# ---------------------------------------------------------------------------


_active_package_data = None
_original_assemble_package = impl.assemble_package
_original_build_method_bases = impl.build_method_bases
_original_configs_from_args = impl._configs_from_args


def assemble_package(cfg: PackageConfig, exp: ExperimentConfig):
    """Capture assembled geometry needed by the BCI boundary residual block."""
    global _active_package_data
    _active_package_data = _original_assemble_package(cfg, exp)
    return _active_package_data


def _source_weighted_port_basis(cfg: PackageConfig, count: int) -> np.ndarray:
    """Select DCT traces using independent chiplet-footprint relevance.

    The score depends on source geometry only, not source power, convection,
    ambient temperature, or a reference full-order solution.  Independent
    chiplet maps avoid relying on the four-chiplet symmetry of the benchmark.
    """
    physical_ports = cfg.physical_ports
    if count <= 0 or count >= physical_ports:
        raise ValueError(
            f"source-aware port count must be in [1, {physical_ports - 1}]"
        )

    full_dct = impl.dct_port_basis(cfg.nx, cfg.ny, physical_ports)
    x_centres = (np.arange(cfg.nx, dtype=np.float64) + 0.5) * (
        cfg.width_mm / cfg.nx
    )
    y_centres = (np.arange(cfg.ny, dtype=np.float64) + 0.5) * (
        cfg.height_mm / cfg.ny
    )
    margin_x = 5.0
    margin_y = 5.0
    positions = (
        (margin_x, margin_y),
        (cfg.width_mm - margin_x - cfg.chiplet_width_mm, margin_y),
        (margin_x, cfg.height_mm - margin_y - cfg.chiplet_height_mm),
        (
            cfg.width_mm - margin_x - cfg.chiplet_width_mm,
            cfg.height_mm - margin_y - cfg.chiplet_height_mm,
        ),
    )

    source_maps = []
    for x0, y0 in positions:
        mask = np.asarray(
            [
                1.0
                if (
                    x0 <= x <= x0 + cfg.chiplet_width_mm
                    and y0 <= y <= y0 + cfg.chiplet_height_mm
                )
                else 0.0
                for x in x_centres
                for y in y_centres
            ],
            dtype=np.float64,
        )
        source_maps.append(mask)
    maps = np.column_stack(source_maps)
    coefficients = np.abs(full_dct.T @ maps)
    relevance = np.linalg.norm(coefficients, axis=1)
    ranked = np.argsort(-relevance, kind="stable")

    # The constant trace is mandatory for exact uniform-temperature states.
    selected = [0]
    selected_set = {0}
    for index in ranked:
        index = int(index)
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
        if len(selected) == count:
            break
    return np.ascontiguousarray(full_dct[:, selected], dtype=np.float64)


def _thin_qr(matrix: np.ndarray, tolerance: float = 1.0e-12) -> np.ndarray:
    """Return a numerically independent orthonormal basis for matrix columns."""
    if matrix.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    q, r = np.linalg.qr(matrix, mode="reduced")
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    threshold = tolerance * max(1.0, float(diagonal.max()))
    rank = int(np.count_nonzero(diagonal > threshold))
    return np.ascontiguousarray(q[:, :rank], dtype=np.float64)


def _randomized_left_basis(
    snapshots: np.ndarray,
    rank: int,
    seed: int,
    oversampling: int = 32,
) -> np.ndarray:
    """Compute a deterministic dominant left singular subspace."""
    rank = min(rank, snapshots.shape[0], snapshots.shape[1])
    if rank <= 0:
        return np.empty((snapshots.shape[0], 0), dtype=np.float64)
    sample_count = min(snapshots.shape[1], rank + oversampling)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((snapshots.shape[1], sample_count))
    sample = snapshots @ omega
    # One power iteration is sufficient for the clear singular-value cluster
    # separation observed in this diffusion operator and improves repeatability.
    sample = snapshots @ (snapshots.T @ sample)
    q = np.linalg.qr(sample, mode="reduced")[0]
    compressed = q.T @ snapshots
    u_hat, _, _ = scipy.linalg.svd(
        compressed,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    return np.ascontiguousarray(q @ u_hat[:, :rank], dtype=np.float64)


def _build_source_aware_bci(
    cfg: PackageConfig,
    core,
    exp: ExperimentConfig,
) -> MethodBasis:
    """Build a BCI residual Krylov basis without embedding a boundary value."""
    if _active_package_data is None:
        raise RuntimeError("package data is unavailable before basis extraction")

    start = time.perf_counter()
    data = _active_package_data
    phi = _source_weighted_port_basis(cfg, exp.port_modes)
    psi = core.constraint_map @ phi

    # Unit top-boundary operator.  Its response span is independent of the
    # eventual convection coefficient h; h is introduced only during projection.
    top_diagonal = np.zeros(core.Kii.shape[0], dtype=np.float64)
    top_diagonal[data.macro_top_local_cells] = data.top_face_area_m2
    unit_boundary = sp.diags(top_diagonal, format="csc")

    static_factor = spla.splu(core.Kii)
    static_residual = -static_factor.solve(unit_boundary @ psi)
    boundary_modes = _thin_qr(static_residual)

    # Multi-point Laplace residuals of the isolated adiabatic macro.  Subtract
    # the Guyan limit and only remove already representable interior-only
    # boundary modes.  Removing the constraint modes themselves would be wrong:
    # their coefficients are tied to nonzero physical-port coordinates.
    port_forcing = core.Kip @ phi
    shift_scale = 0.025 / exp.time_step_s
    shifts = shift_scale * np.asarray(
        (1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0),
        dtype=np.float64,
    )
    residual_blocks = []
    for shift in shifts:
        factor = spla.splu((core.Kii + float(shift) * core.Cii).tocsc())
        frequency_response = factor.solve(-port_forcing)
        residual = frequency_response - psi
        if boundary_modes.shape[1] > 0:
            residual -= boundary_modes @ (boundary_modes.T @ residual)
        residual_blocks.append(residual)

    snapshots = np.hstack(residual_blocks)
    dynamic_rank = min(
        exp.rational_modes,
        snapshots.shape[0] - boundary_modes.shape[1],
        snapshots.shape[1],
    )
    dynamic_modes = _randomized_left_basis(
        snapshots,
        dynamic_rank,
        seed=exp.random_seed,
    )

    interior_only = np.hstack((boundary_modes, dynamic_modes))
    zero_port = np.zeros(
        (cfg.physical_ports, interior_only.shape[1]), dtype=np.float64
    )
    physical_basis = np.hstack((phi, zero_port))
    interior_basis = np.hstack((psi, interior_only))
    basis = np.vstack((physical_basis, interior_basis))
    elapsed = core.preprocess_s + time.perf_counter() - start
    return MethodBasis(
        "source_aware_bci_krylov",
        np.ascontiguousarray(basis, dtype=np.float64),
        np.ascontiguousarray(physical_basis, dtype=np.float64),
        elapsed,
    )


def build_method_bases(
    cfg: PackageConfig,
    core,
    exp: ExperimentConfig,
) -> list[MethodBasis]:
    """Retain four baselines and append the validated BCI residual method."""
    methods = _original_build_method_bases(cfg, core, exp)
    methods.append(_build_source_aware_bci(cfg, core, exp))
    return methods


# ---------------------------------------------------------------------------
# Validated quick/default ranks
# ---------------------------------------------------------------------------


def _configs_from_args(args):
    package, experiment = _original_configs_from_args(args)
    values = asdict(experiment)
    if args.port_modes is None:
        values["port_modes"] = 208 if args.quick else 484
    if args.interior_modes is None:
        # Used as the dominant dynamic-residual rank by the source-aware method.
        values["rational_modes"] = 240 if args.quick else 560
    return package, ExperimentConfig(**values)


# Install overrides into the baseline driver's module namespace.
impl.build_package_model = build_package_model
impl.assemble_package = assemble_package
impl.build_method_bases = build_method_bases
impl._configs_from_args = _configs_from_args


if __name__ == "__main__":
    raise SystemExit(impl.main())
