import argparse

import pyvista as pv


def _pick_scalar(mesh: pv.DataSet) -> str:
    if "Temperature_K" in mesh.cell_data:
        return "Temperature_K"
    if "Temperature" in mesh.cell_data:
        return "Temperature"
    raise KeyError("No Temperature_K or Temperature cell data found")


def visualize(vtu_path: str) -> None:
    mesh = pv.read(vtu_path)
    scalar_name = _pick_scalar(mesh)

    print(f"Loading mesh from {vtu_path}")
    print(mesh)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=scalar_name, cmap="hot", show_edges=True)
    plotter.add_scalar_bar("Temperature (K)")
    plotter.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize temperature field in VTU mesh"
    )
    parser.add_argument("vtu_path", help="Path to result VTU file")
    args = parser.parse_args()
    visualize(args.vtu_path)


if __name__ == "__main__":
    main()
