"""Physical field plots and dependency-free VTK XML hexahedral export."""

from pathlib import Path
from xml.sax.saxutils import quoteattr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from assembly import (
    AMBIENT_C,
    CHIP_RECTS_MM,
    JUNCTION_Z_MM,
    MM,
    SOURCE_NAMES,
)
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from transient import POWER_LEVELS, POWER_TIMES, PROBES_MM, power_at


def _array(stream, name, values, vtk_type="Float64"):
    values = np.asarray(values)
    components = values.shape[1] if values.ndim == 2 else 1
    stream.write(
        f'<DataArray type="{vtk_type}" Name={quoteattr(name)} '
        f'NumberOfComponents="{components}" format="ascii">\n'
    )
    np.savetxt(
        stream,
        values.reshape(-1, components),
        fmt="%.10g" if vtk_type == "Float64" else "%d",
    )
    stream.write("</DataArray>\n")


def write_vtu(path, model, point_fields=None, powers=None, interface_nodes=None):
    """Export the actual FV mesh and nodal fields without changing DOF order."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as out:
        out.write(
            '<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" '
            'version="0.1" byte_order="LittleEndian">\n<UnstructuredGrid>\n'
        )
        out.write(
            f'<Piece NumberOfPoints="{model.N}" '
            f'NumberOfCells="{len(model.cells)}">\n<Points>\n'
        )
        _array(out, "coordinates_m", model.points)
        out.write("</Points>\n<Cells>\n")
        _array(out, "connectivity", model.cells.ravel(), "Int64")
        _array(out, "offsets", 8 * np.arange(1, len(model.cells) + 1), "Int64")
        _array(out, "types", np.full(len(model.cells), 12), "UInt8")
        out.write("</Cells>\n<PointData>\n")
        interface = np.zeros(model.N, dtype=np.uint8)
        interface[
            np.arange(model.m) if interface_nodes is None else interface_nodes
        ] = 1
        _array(out, "interface_node", interface, "UInt8")
        _array(out, "heat_capacity_J_K", model.capacity)
        for name, values in (point_fields or {}).items():
            if np.asarray(values).shape != (model.N,):
                raise ValueError(f"{name}: expected one scalar per mesh vertex")
            _array(out, name, values)
        out.write("</PointData>\n<CellData>\n")
        _array(out, "layer_id", model.cell_layer, "Int32")
        _array(out, "chip_footprint_mask", model.chip_fraction)
        _array(out, "conductivity_xyz_W_mK", model.cell_conductivity)
        _array(out, "rho_cp_J_m3K", model.cell_rhoc)
        for i, name in enumerate(SOURCE_NAMES):
            _array(out, f"{name}_unit_source_m-3", model.source_density[:, i])
        if powers is not None:
            _array(out, "power_density_W_m3", model.source_density @ powers)
        out.write("</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n")
    temporary.replace(path)


def _write_assembly(path, component_file, cooler_file):
    path.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="vtkMultiBlockDataSet" '
        'version="1.0" byte_order="LittleEndian">\n<vtkMultiBlockDataSet>\n'
        f'<DataSet index="0" name="component" file="{component_file}"/>\n'
        f'<DataSet index="1" name="TIM_and_lid" file="{cooler_file}"/>\n'
        "</vtkMultiBlockDataSet>\n</VTKFile>\n",
        encoding="utf-8",
    )


def _fields(fom, rom):
    return {
        "FOM_temperature_C": AMBIENT_C + fom,
        "ROM_temperature_C": AMBIENT_C + rom,
        "ROM_minus_FOM_K": rom - fom,
    }


def export_fields(directory, model, result):
    directory.mkdir(parents=True, exist_ok=True)
    write_vtu(directory / "model.vtu", model)
    cooler = result.get("cooler")
    if cooler is not None:
        interface = np.arange(cooler.m)
        write_vtu(
            directory / "cooler_steady.vtu",
            cooler,
            _fields(*result["cooler_steady"]),
            interface_nodes=interface,
        )
        cooler_datasets = []
        for i, (t, f, r) in enumerate(result["cooler_snapshots"]):
            name = f"cooler_{i:03d}.vtu"
            write_vtu(
                directory / name, cooler, _fields(f, r), interface_nodes=interface
            )
            cooler_datasets.append(
                f'<DataSet timestep="{t:.12g}" group="" part="0" file="{name}"/>'
            )
        (directory / "cooler.pvd").write_text(
            '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" '
            'byte_order="LittleEndian"><Collection>\n'
            + "\n".join(cooler_datasets)
            + "\n</Collection></VTKFile>\n",
            encoding="utf-8",
        )
    datasets = []
    for i, (t, f, r) in enumerate(result["snapshots"]):
        name = f"transient_{i:03d}.vtu"
        # Fields at a jump are continuous and correspond to the preceding
        # interval's load; export that load rather than the just-switched value.
        write_vtu(directory / name, model, _fields(f, r), power_at(t - 1e-12))
        datasets.append(
            f'<DataSet timestep="{t:.12g}" group="" part="0" file="{name}"/>'
        )
    (directory / "transient.pvd").write_text(
        '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" '
        'byte_order="LittleEndian">\n<Collection>\n'
        + "\n".join(datasets)
        + "\n</Collection>\n</VTKFile>\n",
        encoding="utf-8",
    )
    write_vtu(
        directory / "steady.vtu",
        model,
        _fields(*result["steady"]),
        POWER_LEVELS[POWER_LEVELS.sum(axis=1).argmax()],
    )

    if cooler is not None:
        assembly_datasets = []
        for i, (t, _, _) in enumerate(result["snapshots"]):
            name = f"assembly_{i:03d}.vtm"
            _write_assembly(
                directory / name, f"transient_{i:03d}.vtu", f"cooler_{i:03d}.vtu"
            )
            assembly_datasets.append(
                f'<DataSet timestep="{t:.12g}" group="" part="0" file="{name}"/>'
            )
        (directory / "assembly.pvd").write_text(
            '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" '
            'byte_order="LittleEndian"><Collection>\n'
            + "\n".join(assembly_datasets)
            + "\n</Collection></VTKFile>\n",
            encoding="utf-8",
        )
        _write_assembly(
            directory / "steady_assembly.vtm", "steady.vtu", "cooler_steady.vtu"
        )


def _save(fig, path):
    fig.savefig(path, dpi=180)
    plt.close(fig)


def section_cells(model, dim, coordinate):
    vertices = model.points[model.cells] / MM
    lower, upper = vertices[:, :, dim].min(axis=1), vertices[:, :, dim].max(axis=1)
    # Half-open cells avoid drawing both sides of a material interface.
    selected = (lower <= coordinate + 1e-10) & (upper > coordinate + 1e-10)
    return vertices, selected


def plot_model(model, path, cooler):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    handles = []
    for region, offset in ((model, 0), (cooler, len(model.layer_specs))):
        vertices, selected = section_cells(region, 1, 9.0)
        for layer, spec in enumerate(region.layer_specs):
            color = plt.cm.tab10(layer + offset)
            ids = selected & (region.cell_layer == layer)
            polygons = []
            for cell in vertices[ids]:
                x0, z0 = cell[:, [0, 2]].min(axis=0)
                x1, z1 = cell[:, [0, 2]].max(axis=0)
                polygons.append([(x0, z0), (x1, z0), (x1, z1), (x0, z1)])
            axes[0].add_collection(
                PolyCollection(polygons, facecolor=color, edgecolor="none")
            )
            handles.append(Rectangle((0, 0), 1, 1, color=color, label=spec[0]))
    axes[0].set(
        xlim=(0, 24),
        ylim=(0, cooler.axes[2][-1] / MM + 0.05),
        xlabel="x [mm]",
        ylabel="z [mm]",
        title="Full assembly at y = 9 mm; cavities are empty",
    )
    axes[0].legend(
        handles=handles,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )
    for name, rect in zip(SOURCE_NAMES, CHIP_RECTS_MM):
        x0, y0, x1, y1 = rect
        axes[1].add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#9abbd8", edgecolor="k")
        )
        axes[1].text((x0 + x1) / 2, (y0 + y1) / 2, name, ha="center")
    axes[1].set(
        xlim=(0, 24),
        ylim=(0, 24),
        xlabel="x [mm]",
        ylabel="y [mm]",
        title="Silicon / TIM footprints; full top interfaces retained",
    )
    axes[1].set_aspect("equal")
    fig.suptitle(
        f"Component {model.N} DOFs / external FOM {cooler.N} DOFs / "
        f"{model.m} shared ports"
    )
    _save(fig, path)


def plot_transient(result, path):
    t = result["times"]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        constrained_layout=True,
        height_ratios=[1, 2],
    )
    for q, name in enumerate(SOURCE_NAMES):
        axes[0].step(
            np.r_[POWER_TIMES, t[-1]],
            np.r_[POWER_LEVELS[:, q], POWER_LEVELS[-1, q]],
            where="post",
            label=name,
        )
    axes[0].set_ylabel("power [W]")
    axes[0].legend(ncol=2)
    for q, name in enumerate(result.get("probe_names", PROBES_MM)):
        color = plt.cm.tab10(q)
        axes[1].plot(
            t, AMBIENT_C + result["FOM_probes"][:, q], color=color, label=f"{name} FOM"
        )
        axes[1].plot(
            t,
            AMBIENT_C + result["ROM_probes"][:, q],
            color=color,
            ls="--",
            label=f"{name} ROM",
        )
    axes[1].set(xlabel="time [s]", ylabel="temperature [°C]")
    axes[1].legend(ncol=2, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlim(0, t[-1])
    _save(fig, path)


def _slice(region, field, dim, coordinate):
    vertices, selected = section_cells(region, dim, coordinate)
    ids = np.flatnonzero(selected)
    remaining = [d for d in range(3) if d != dim]
    polygons, values = [], []
    for index in ids:
        cell = vertices[index]
        lo, hi = cell[:, remaining].min(axis=0), cell[:, remaining].max(axis=0)
        polygons.append(
            [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
        )
        # Trilinear field averaged over this cut rectangle, without crossing a cavity.
        lower, upper = cell[:, dim].min(), cell[:, dim].max()
        alpha = (coordinate - lower) / (upper - lower)
        nodal = field[region.cells[index]]
        values.append(
            (1 - alpha) * nodal[np.isclose(cell[:, dim], lower)].mean()
            + alpha * nodal[np.isclose(cell[:, dim], upper)].mean()
        )
    return polygons, np.asarray(values)


def plot_fields(model, fom, rom, title, path, cooler, cooler_fields):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    vmin = AMBIENT_C + min(fom.min(), rom.min(), *(a.min() for a in cooler_fields))
    vmax = AMBIENT_C + max(fom.max(), rom.max(), *(a.max() for a in cooler_fields))
    for row, dim, coordinate in ((0, 2, JUNCTION_Z_MM), (1, 1, 9.0)):
        regions = [(model, fom, rom)]
        if row == 1:
            regions.append((cooler, *cooler_fields))
        slices = [
            (_slice(region, f, dim, coordinate), _slice(region, r, dim, coordinate))
            for region, f, r in regions
        ]
        polygons = [polygon for a, _ in slices for polygon in a[0]]
        f = np.concatenate([a[1] for a, _ in slices])
        r = np.concatenate([b[1] for _, b in slices])
        for col, values in enumerate((AMBIENT_C + f, AMBIENT_C + r, r - f)):
            limit = max(float(np.max(np.abs(values))), 1e-9)
            collection = PolyCollection(
                polygons,
                array=values,
                cmap="coolwarm" if col == 2 else "inferno",
                edgecolor="none",
            )
            collection.set_clim((-limit, limit) if col == 2 else (vmin, vmax))
            axis = axes[row, col]
            axis.add_collection(collection)
            axis.set(
                xlim=(0, 24),
                ylim=(0, 24 if row == 0 else cooler.axes[2][-1] / MM),
                xlabel="x [mm]",
                ylabel="y [mm]" if row == 0 else "z [mm]",
                title=("FOM", "ROM + external FOM", "ROM - FOM")[col]
                + (": junction" if row == 0 else ": complete assembly"),
            )
            if row == 0:
                axis.set_aspect("equal")
            fig.colorbar(
                collection,
                ax=axis,
                label="error [K]" if col == 2 else "temperature [°C]",
            )
    fig.suptitle(title + "; empty areas contain no cells")
    _save(fig, path)


def write_outputs(directory, model, result):
    directory.mkdir(parents=True, exist_ok=True)
    plot_model(model, directory / "package.png", result["cooler"])
    plot_transient(result, directory / "transient_probes.png")
    t, f, r = max(result["snapshots"], key=lambda item: item[1].max())
    plot_fields(
        model,
        f,
        r,
        f"{model.material}: hottest stored transient snapshot at t = {t:g} s",
        directory / "transient_fields.png",
        result["cooler"],
        next((cf, cr) for ct, cf, cr in result["cooler_snapshots"] if ct == t),
    )
    peak_power = POWER_LEVELS.sum(axis=1).max()
    plot_fields(
        model,
        *result["steady"],
        f"{model.material}: sustained {peak_power:g} W steady state",
        directory / "steady_fields.png",
        result["cooler"],
        result["cooler_steady"],
    )
    np.savetxt(
        directory / "probe_temperatures.csv",
        np.column_stack(
            [
                result["times"],
                power_at(result["times"]),
                AMBIENT_C + result["FOM_probes"],
                AMBIENT_C + result["ROM_probes"],
            ]
        ),
        delimiter=",",
        header=",".join(
            ["time_s"]
            + [f"{s}_W" for s in SOURCE_NAMES]
            + [f"FOM_{s}_C" for s in result["probe_names"]]
            + [f"ROM_{s}_C" for s in result["probe_names"]]
        ),
        comments="",
    )
    export_fields(directory / "vtk", model, result)


def plot_connections(nominal, connected, path):
    """Overlay independent FOM/ROM trajectories under two external systems."""
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), constrained_layout=True, sharex=True
    )
    for axis, data, label in zip(
        axes,
        (nominal, connected),
        (
            "Full TIM/lid: nominal cooling",
            "Full TIM/lid: half lid HTC; same component ROM",
        ),
    ):
        for q, name in enumerate(data["probe_names"]):
            color = plt.cm.tab10(q)
            axis.plot(
                data["times"],
                AMBIENT_C + data["FOM_probes"][:, q],
                color=color,
                label=f"{name} FOM",
            )
            axis.plot(
                data["times"],
                AMBIENT_C + data["ROM_probes"][:, q],
                color=color,
                ls="--",
                label=f"{name} ROM",
            )
        error = np.max(np.abs(data["ROM_probes"] - data["FOM_probes"]))
        axis.set(
            title=f"{label}: maximum probe error {error:.3g} K",
            ylabel="temperature [°C]",
        )
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("time [s]")
    _save(fig, path)
