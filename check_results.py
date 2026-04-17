import meshio
import numpy as np
import sys

def check(vtu_path):
    mesh = meshio.read(vtu_path)
    temp = mesh.cell_data.get('Temperature')
    if temp is None:
        print("Error: No Temperature data found!")
        return
    
    # temp is a list of arrays (one for each cell block)
    # Since we saved it as a single array for all cells
    t_data = temp[0]
    print(f"Results for {vtu_path}:")
    print(f"  Min Temperature: {np.min(t_data):.2f} K")
    print(f"  Max Temperature: {np.max(t_data):.2f} K")
    print(f"  Mean Temperature: {np.mean(t_data):.2f} K")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_results.py <vtu_path>")
    else:
        check(sys.argv[1])
