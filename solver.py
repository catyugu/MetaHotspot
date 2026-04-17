import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import meshio
import toml
import os

class FVMSolver:
    def __init__(self, config_path, mesh_path):
        self.config = toml.load(config_path)
        self.mesh = meshio.read(mesh_path)
        self.materials = self.config['materials']
        self.cells = []
        self.node_to_cells = {}
        self._prepare_mesh()

    def _prepare_mesh(self):
        print("Preparing mesh data...")
        hex_data = self.mesh.cells_dict.get('hexahedron')
        if hex_data is None: raise ValueError("No hexahedral cells found!")
        physical_tags = self.mesh.cell_data_dict.get('gmsh:physical', {}).get('hexahedron')
        points = self.mesh.points
        
        for i, node_indices in enumerate(hex_data):
            c_pts = points[node_indices]
            min_pts = np.min(c_pts, axis=0); max_pts = np.max(c_pts, axis=0)
            dims = max_pts - min_pts
            if np.any(dims <= 0): dims = np.maximum(dims, 1e-12)
            
            tag = physical_tags[i] if physical_tags is not None else 1
            unit = self.config['power_units'][tag - 1]
            mat_name = unit['material']
            k = self.materials[mat_name]['k']
            
            cell_info = {
                'id': i, 'center': (min_pts + max_pts) / 2, 'dims': dims,
                'vol': dims[0]*dims[1]*dims[2], 'k': k, 'name': unit['name'],
                'layer_id': unit.get('layer_id', 0), 'nodes': set(node_indices), 'tag': tag
            }
            self.cells.append(cell_info)
            for nid in node_indices:
                if nid not in self.node_to_cells: self.node_to_cells[nid] = []
                self.node_to_cells[nid].append(i)

    def load_power(self, ptrace_path):
        if not ptrace_path:
            self.power_map = {}
            return
        print(f"Loading power from {ptrace_path}...")
        with open(ptrace_path, 'r') as f:
            lines = f.readlines()
            header = lines[0].strip().split()
            data = [[float(x) for x in l.strip().split()] for l in lines[1:] if l.strip()]
            avg_power = np.mean(data, axis=0)
            self.power_map = dict(zip(header, avg_power))

    def solve_steady(self, ptrace_path=None):
        self.load_power(ptrace_path)
        n = len(self.cells)
        rows, cols, data = [], [], []
        b = np.zeros(n)
        group_cell_counts = {}
        for cell in self.cells:
            tag = cell['tag']
            group_cell_counts[tag] = group_cell_counts.get(tag, 0) + 1

        print(f"Assembling matrix for {n} cells...")
        # 1. Internal Conduction
        for i, cell in enumerate(self.cells):
            candidates = set()
            for nid in cell['nodes']: candidates.update(self.node_to_cells[nid])
            for j in candidates:
                if i >= j: continue
                neighbor = self.cells[j]
                shared = cell['nodes'].intersection(neighbor['nodes'])
                if len(shared) >= 4:
                    dist = neighbor['center'] - cell['center']
                    axis = np.argmax(np.abs(dist))
                    area = cell['dims'][(axis+1)%3] * cell['dims'][(axis+2)%3]
                    R_i = (cell['dims'][axis] / 2) / (cell['k'] * area)
                    R_j = (neighbor['dims'][axis] / 2) / (neighbor['k'] * area)
                    G = 1.0 / (R_i + R_j)
                    rows.extend([i, j, i, j]); cols.extend([i, j, j, i]); data.extend([-G, -G, G, G])

            b[i] = -self.power_map.get(cell['name'], 0.0) / group_cell_counts[cell['tag']]
            
        # 2. Generic Boundary Conditions from Config
        print("Applying boundary conditions...")
        for bc in self.config.get('boundary_conditions', []):
            bc_type = bc.get('type')
            selection = bc.get('selection', [])
            
            if bc_type == 'convection':
                h = bc.get('h', 0.0)
                T_inf = bc.get('T_inf', 293.15)
                
                for i in [idx for idx, c in enumerate(self.cells) if c['tag'] in selection]:
                    cell = self.cells[i]
                    # Identify external faces of this cell
                    # A face is external if its nodes are shared with NO other cell
                    # Let's simplify: check the 6 directions for neighbors
                    for axis in range(3):
                        for direction in [-1, 1]:
                            is_external = True
                            # Search for neighbor in this direction
                            target_center = cell['center'].copy()
                            target_center[axis] += direction * cell['dims'][axis]
                            
                            # If no neighbor is found within a small tolerance, it's an external face
                            for nid in cell['nodes']:
                                for neighbor_idx in self.node_to_cells[nid]:
                                    if neighbor_idx == i: continue
                                    neighbor = self.cells[neighbor_idx]
                                    if np.linalg.norm(neighbor['center'] - target_center) < 1e-7:
                                        is_external = False
                                        break
                                if not is_external: break
                            
                            if is_external:
                                area = cell['dims'][(axis+1)%3] * cell['dims'][(axis+2)%3]
                                G_bc = h * area
                                rows.append(i); cols.append(i); data.append(-G_bc)
                                b[i] -= G_bc * T_inf

        G_mat = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        print("Solving linear system...")
        T = splinalg.spsolve(G_mat, b)
        return T

    def save_results(self, T, vtu_path):
        self.mesh.cell_data['Temperature'] = [np.array(T)]
        self.mesh.write(vtu_path)
        print(f"3D Results saved to {vtu_path}")
        
        # Save per-layer .steady files
        output_dir = os.path.dirname(vtu_path)
        layers = sorted(list(set(c['layer_id'] for c in self.cells)))
        for layer_id in layers:
            layer_results = {} # unit_name -> sum_temp, count
            for i, cell in enumerate(self.cells):
                if cell['layer_id'] == layer_id:
                    name = cell['name']
                    if name not in layer_results: layer_results[name] = []
                    layer_results[name].append(T[i])
            
            steady_path = os.path.join(output_dir, f"layer_{layer_id}.steady")
            with open(steady_path, 'w') as f:
                for name, temps in layer_results.items():
                    f.write(f"{name}\t{np.mean(temps):.2f}\n")
            print(f"Layer {layer_id} results saved to {steady_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4: print("Usage: python solver.py <config> <mesh> <output_vtu> [ptrace]")
    else:
        solver = FVMSolver(sys.argv[1], sys.argv[2])
        T = solver.solve_steady(sys.argv[4] if len(sys.argv) > 4 else None)
        solver.save_results(T, sys.argv[3])
