#!/usr/bin/env python3
"""Entry point for the transient BCI-ROM benchmark.

The implementation is kept in ``_macromodel_demo_impl``.  This entry module
corrects the authoring-layer insertion order to match MetaHotspot's top-first
layer compiler contract, then delegates to the research driver.
"""

from __future__ import annotations

import _macromodel_demo_impl as impl

metahotspot = impl.metahotspot
GeometryOp = impl.GeometryOp
LengthUnit = impl.LengthUnit
Study = impl.Study
_axis_vertices = impl._axis_vertices
_layered_z_vertices = impl._layered_z_vertices
_add_materials = impl._add_materials
_add_full_rect = impl._add_full_rect
_chiplet_heat_source = impl._chiplet_heat_source


def build_package_model(
    cfg: PackageConfig,
    *,
    include_macro: bool,
    study: Study = Study.STEADY,
    duration_s: float = 0.0,
    output_interval_s: float = 0.0,
):
    """Build a heterogeneous package using only the public scripting API."""
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

    # IMPORTANT: the compiler stacks authoring layers in reverse insertion
    # order: the first added layer occupies the highest z interval.  Register
    # the physical package from top to bottom so the mesh remains ordered from
    # substrate (z=0) to cold plate (z=max).
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

    # Mold background with four active silicon chiplets.  This is the top
    # layer of the independently compiled detailed component.
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

    # Underfill with an 8x8 copper bump array.  Later blocks within the same
    # layer override the underfill background in cells covered by a bump.
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

    # Organic substrate is inserted last and therefore occupies the lowest z.
    substrate_layer = model.add_layer(f"{cfg.substrate_mm:.17g}")
    substrate = model.add_block(substrate_layer, "organic")
    _add_full_rect(model, substrate, cfg)

    # Adiabatic isolation is essential: external boundary operators are added
    # after extraction and may be changed without recomputing the basis.
    model.set_default_neumann("0")
    return model


impl.build_package_model = build_package_model


if __name__ == "__main__":
    raise SystemExit(impl.main())
