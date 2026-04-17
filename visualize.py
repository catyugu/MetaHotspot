import pyvista as pv
import sys

def visualize(vtu_path):
    mesh = pv.read(vtu_path)
    print(f"Loading mesh from {vtu_path}")
    print(mesh)
    
    # Plot
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='Temperature', cmap='hot', show_edges=True)
    plotter.add_scalar_bar('Temperature (K)')
    plotter.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <vtu_path>")
    else:
        visualize(sys.argv[1])
