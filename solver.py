import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import meshio
import toml
import os

def get_overlap_area(b1, b2, axis):
    """Calculate overlap area between two 3D boxes along a specific axis face"""
    # Plane indices based on normal axis
    idx = [i for i in range(3) if i != axis]
    # Intersect 2D rectangles in the face plane
    inter_min = np.maximum(b1[idx], b2[idx])
    inter_max = np.minimum(b1[[i+3 for i in idx]], b2[[i+3 for i in idx]])
    dims = inter_max - inter_min
    return np.prod(dims) if np.all(dims > 0) else 0.0

class FVMSolver:
    def __init__(self, config_path):
        self.base_dir = os.path.dirname(config_path)
        self.config = toml.load(config_path)
        self.mesh = meshio.read(os.path.join(self.base_dir, self.config.get('mesh_file_path', 'mesh.msh')))
        self.materials = self.config['materials']
        self.tag_to_mat = {t: m for m, tags in self.config.get('domain_material_assignment', {}).items() for t in tags}
        self.cells = []
        self._prepare_mesh()

    def _prepare_mesh(self):
        print("[INFO] Preparing mesh data...")
        hex_data = self.mesh.cells_dict.get('hexahedron')
        physical_tags = self.mesh.cell_data_dict.get('gmsh:physical', {}).get('hexahedron')
        points = self.mesh.points
        for i, nodes in enumerate(hex_data):
            p = points[nodes]; p_min, p_max = np.min(p, axis=0), np.max(p, axis=0)
            tag = physical_tags[i] if physical_tags is not None else -1
            mat = self.tag_to_mat.get(tag, 'silicon')
            self.cells.append({
                'id': i, 'center': (p_min+p_max)/2, 'dims': p_max-p_min, 'box': np.concatenate([p_min, p_max]),
                'k': self.materials[mat]['k'], 'cp': self.materials[mat]['cp'], 'tag': tag, 'vol': np.prod(p_max-p_min)
            })

    def assemble_g_matrix(self):
        n = len(self.cells)
        rows, cols, data = [], [], []
        print(f"[INFO] Building non-conformal G matrix ({n} cells)...")
        
        # Optimization: group cells by layers to avoid O(N^2)
        z_coords = sorted(list(set(c['center'][2] for c in self.cells)))
        layer_cells = {z: [c for c in self.cells if c['center'][2] == z] for z in z_coords}
        
        for idx, z in enumerate(z_coords):
            cells = layer_cells[z]
            # 1. Horizontal Neighbors (within layer)
            for i, c1 in enumerate(cells):
                for j in range(i+1, len(cells)):
                    c2 = cells[j]
                    # Check X-adjacency
                    if abs(c1['center'][1]-c2['center'][1]) < 1e-7: # Same Y
                        if abs(abs(c1['center'][0]-c2['center'][0]) - (c1['dims'][0]+c2['dims'][0])/2) < 1e-7:
                            area = c1['dims'][1] * c1['dims'][2]
                            G = 1.0 / ((c1['dims'][0]/2)/(c1['k']*area) + (c2['dims'][0]/2)/(c2['k']*area))
                            rows.extend([c1['id'], c2['id'], c1['id'], c2['id']]); cols.extend([c1['id'], c2['id'], c2['id'], c1['id']]); data.extend([-G, -G, G, G])
                    # Check Y-adjacency
                    if abs(c1['center'][0]-c2['center'][0]) < 1e-7: # Same X
                        if abs(abs(c1['center'][1]-c2['center'][1]) - (c1['dims'][1]+c2['dims'][1])/2) < 1e-7:
                            area = c1['dims'][0] * c1['dims'][2]
                            G = 1.0 / ((c1['dims'][1]/2)/(c1['k']*area) + (c2['dims'][1]/2)/(c2['k']*area))
                            rows.extend([c1['id'], c2['id'], c1['id'], c2['id']]); cols.extend([c1['id'], c2['id'], c2['id'], c1['id']]); data.extend([-G, -G, G, G])

            # 2. Vertical Neighbors (cross layer)
            if idx < len(z_coords) - 1:
                next_cells = layer_cells[z_coords[idx+1]]
                for c1 in cells:
                    # Check if c1 top touches next layer bottom
                    if abs(c1['box'][5] - (z_coords[idx+1] - next_cells[0]['dims'][2]/2)) < 1e-8:
                        for c2 in next_cells:
                            area = get_overlap_area(c1['box'], c2['box'], 2)
                            if area > 1e-11:
                                G = 1.0 / ((c1['dims'][2]/2)/(c1['k']*area) + (c2['dims'][2]/2)/(c2['k']*area))
                                rows.extend([c1['id'], c2['id'], c1['id'], c2['id']]); cols.extend([c1['id'], c2['id'], c2['id'], c1['id']]); data.extend([-G, -G, G, G])
        
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    def solve(self):
        ptrace_path = os.path.join(self.base_dir, self.config.get('ptrace_file_path', ''))
        ptrace_steps = []
        if os.path.exists(ptrace_path):
            with open(ptrace_path, 'r') as f:
                header = f.readline().split()
                ptrace_steps = [dict(zip(header, [float(x) for x in l.split()])) for l in f if l.strip()]

        G = self.assemble_g_matrix(); n = len(self.cells); b_bc = np.zeros(n); r_bc, c_bc, d_bc = [], [], []
        
        # Convection BC
        for bc in self.config.get('boundary_conditions', []):
            if bc['type'] == 'convection':
                h, T_inf, sel = bc['h'], bc['T_inf'], bc['selection']
                z_max = max(c['box'][5] for c in self.cells)
                for c in [c for c in self.cells if c['tag'] in sel]:
                    if abs(c['box'][5] - z_max) < 1e-7:
                        area = c['dims'][0] * c['dims'][1]; G_bc = h * area
                        r_bc.append(c['id']); c_bc.append(c['id']); d_bc.append(-G_bc); b_bc[c['id']] += G_bc * T_inf
        G += sp.csr_matrix((d_bc, (r_bc, c_bc)), shape=(n, n))

        if self.config.get('simulation_type') == 'steady':
            print("[SIM] Solving Steady State...")
            p_avg = {u['name']: np.mean([s[u['name']] for s in ptrace_steps]) for u in self.config['power_units']} if ptrace_steps else {}
            b = b_bc.copy()
            for c in self.cells:
                for u in self.config['power_units']:
                    inter = np.prod(np.maximum(0, np.minimum(c['box'][3:], [u['lx']+u['dx'],u['ly']+u['dy'],u['lz']+u['dz']]) - np.maximum(c['box'][:3], [u['lx'],u['ly'],u['lz']])))
                    if inter > 1e-15: b[c['id']] += (inter / (u['dx']*u['dy']*u['dz'])) * p_avg.get(u['name'], 0.0)
            T = splinalg.spsolve(-G, b)
            print(f"[RESULT] T_min={np.min(T):.2f} K, T_max={np.max(T):.2f} K")
            self.save(T, "result.vtu")
        else:
            print("[SIM] Solving Transient...")
            dt = self.config.get('timestep', 0.01)
            C = sp.diags([c['cp'] * c['vol'] for c in self.cells])
            # Start from Steady State if not specified
            T = np.full(n, self.config.get('init_temperature', 318.15))
            A_mat = C/dt - G
            for step, p_step in enumerate(ptrace_steps):
                b = (C/dt) @ T + b_bc
                for c in self.cells:
                    for u in self.config['power_units']:
                        inter = np.prod(np.maximum(0, np.minimum(c['box'][3:], [u['lx']+u['dx'],u['ly']+u['dy'],u['lz']+u['dz']]) - np.maximum(c['box'][:3], [u['lx'],u['ly'],u['lz']])))
                        if inter > 1e-15: b[c['id']] += (inter / (u['dx']*u['dy']*u['dz'])) * p_step.get(u['name'], 0.0)
                T = splinalg.spsolve(A_mat, b)
                if step % 10 == 0 or step == len(ptrace_steps)-1:
                    print(f"[STEP {step:4d}] T_min={np.min(T):.2f} K, T_max={np.max(T):.2f} K")
            self.save(T, "transient_result.vtu")

    def save(self, T, name):
        self.mesh.cell_sets = {}; self.mesh.cell_data = {'Temperature': [np.array(T)]}
        self.mesh.write(os.path.join(self.base_dir, name))
        print(f"[FILE] Results saved to {name}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2: print("Usage: python solver.py <config.toml>")
    else: FVMSolver(sys.argv[1]).solve()
