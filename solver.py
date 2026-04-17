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
        self.tag_to_mat = {}
        for mat_name, tags in self.config.get('domain_material_assignment', {}).items():
            for tag in tags: self.tag_to_mat[tag] = mat_name
        
        self.cells = []
        self.node_to_cells = {}
        self._prepare_mesh()

    def _prepare_mesh(self):
        print("Preparing mesh data...")
        hex_data = self.mesh.cells_dict.get('hexahedron')
        physical_tags = self.mesh.cell_data_dict.get('gmsh:physical', {}).get('hexahedron')
        points = self.mesh.points
        
        for i, node_indices in enumerate(hex_data):
            c_pts = points[node_indices]
            min_pts, max_pts = np.min(c_pts, axis=0), np.max(c_pts, axis=0)
            dims = max_pts - min_pts
            if np.any(dims <= 0): dims = np.maximum(dims, 1e-12)
            
            tag = physical_tags[i] if physical_tags is not None else -1
            mat_name = self.tag_to_mat.get(tag, 'silicon')
            k = self.materials[mat_name]['k']
            
            # Find name from power_units if it exists
            name = f"cell_{i}_tag_{tag}"
            layer_id = tag
            for u in self.config.get('power_units', []):
                if u.get('domain_id') == tag:
                    name = u['name']
                    layer_id = u.get('layer_id', 0)
                    break
            
            cell_info = {
                'id': i, 'center': (min_pts + max_pts) / 2, 'dims': dims,
                'k': k, 'name': name, 'layer_id': layer_id, 'nodes': set(node_indices), 'tag': tag
            }
            self.cells.append(cell_info)
            for nid in node_indices:
                if nid not in self.node_to_cells: self.node_to_cells[nid] = []
                self.node_to_cells[nid].append(i)

    def solve_steady(self, ptrace_path=None):
        # 1. Load Power
        power_map = {}
        if ptrace_path:
            with open(ptrace_path, 'r') as f:
                lines = f.readlines()
                header = lines[0].strip().split()
                data = [[float(x) for x in l.strip().split()] for l in lines[1:] if l.strip()]
                avg_power = np.mean(data, axis=0)
                power_map = dict(zip(header, avg_power))

        n = len(self.cells)
        rows, cols, data, b = [], [], [], np.zeros(n)
        
        # Power source density calculation
        tag_cell_counts = {}
        for c in self.cells: tag_cell_counts[c['tag']] = tag_cell_counts.get(c['tag'], 0) + 1

        print(f"Assembling matrix for {n} cells...")
        for i, cell in enumerate(self.cells):
            # Internal Conduction
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
                    R_i, R_j = (cell['dims'][axis]/2)/(cell['k']*area), (neighbor['dims'][axis]/2)/(neighbor['k']*area)
                    G = 1.0 / (R_i + R_j)
                    rows.extend([i, j, i, j]); cols.extend([i, j, j, i]); data.extend([-G, -G, G, G])
            
            # Apply power source ONLY if in power_units
            if cell['name'] in power_map:
                b[i] = -power_map[cell['name']] / tag_cell_counts[cell['tag']]
            
        # 2. Boundary Conditions
        for bc in self.config.get('boundary_conditions', []):
            if bc.get('type') == 'convection':
                h, T_inf, selection = bc.get('h', 0.0), bc.get('T_inf', 293.15), bc.get('selection', [])
                for i in [idx for idx, c in enumerate(self.cells) if c['tag'] in selection]:
                    cell = self.cells[i]
                    for axis in range(3):
                        for direction in [-1, 1]:
                            is_ext = True; target = cell['center'].copy(); target[axis] += direction * cell['dims'][axis]
                            for nid in cell['nodes']:
                                for nj in self.node_to_cells[nid]:
                                    if nj != i and np.linalg.norm(self.cells[nj]['center']-target) < 1e-7:
                                        is_ext = False; break
                                if not is_ext: break
                            if is_ext:
                                area = cell['dims'][(axis+1)%3] * cell['dims'][(axis+2)%3]
                                G_bc = h * area; rows.append(i); cols.append(i); data.append(-G_bc); b[i] -= G_bc * T_inf

        print("Solving...")
        T = splinalg.spsolve(sp.csr_matrix((data, (rows, cols)), shape=(n, n)), b)
        return T

    def save_results(self, T, vtu_path):
        self.mesh.cell_data['Temperature'] = [np.array(T)]
        self.mesh.write(vtu_path)
        # Layer-wise results (filtering for chip layers only)
        output_dir = os.path.dirname(vtu_path)
        active_tags = set(u.get('domain_id') for u in self.config.get('power_units', []))
        for tid in sorted(list(set(c['tag'] for c in self.cells))):
            if tid in active_tags:
                layer_res = {c['name']: [] for c in self.cells if c['tag'] == tid}
                for i, c in enumerate(self.cells):
                    if c['tag'] == tid: layer_res[c['name']].append(T[i])
                with open(os.path.join(output_dir, f"domain_{tid}.steady"), 'w') as f:
                    for n, ts in layer_res.items(): f.write(f"{n}\t{np.mean(ts):.2f}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4: print("Usage: python solver.py <config> <mesh> <vtu> [ptrace]")
    else:
        s = FVMSolver(sys.argv[1], sys.argv[2]); T = s.solve_steady(sys.argv[4] if len(sys.argv) > 4 else None); s.save_results(T, sys.argv[3])
