import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import meshio
import toml
import os

def get_intersection_volume(box1, box2):
    inter_min = np.maximum(box1[:3], box2[:3])
    inter_max = np.minimum(box1[3:], box2[3:])
    diff = inter_max - inter_min
    if np.any(diff <= 0): return 0.0
    return np.prod(diff)

class FVMSolver:
    def __init__(self, config_path):
        self.base_dir = os.path.dirname(config_path)
        self.config = toml.load(config_path)
        
        # Load mesh from config path
        mesh_full_path = os.path.join(self.base_dir, self.config.get('mesh_file_path', 'mesh.msh'))
        self.mesh = meshio.read(mesh_full_path)
        
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
            min_p, max_p = np.min(c_pts, axis=0), np.max(c_pts, axis=0)
            dims = max_p - min_p
            if np.any(dims <= 0): dims = np.maximum(dims, 1e-12)
            tag = physical_tags[i] if physical_tags is not None else -1
            mat_name = self.tag_to_mat.get(tag, 'silicon')
            
            cell_box = np.concatenate([min_p, max_p])
            # Material Override check
            for u in self.config.get('power_units', []):
                if 'material' in u:
                    u_box = [u['lx'], u['ly'], u['lz'], u['lx']+u['dx'], u['ly']+u['dy'], u['lz']+u['dz']]
                    if get_intersection_volume(cell_box, u_box) > 0.5 * np.prod(dims):
                        mat_name = u['material']; break

            cell_info = {
                'id': i, 'center': (min_p + max_p) / 2, 'dims': dims, 'box': cell_box,
                'k': self.materials[mat_name]['k'], 'tag': tag, 'nodes': set(node_indices)
            }
            self.cells.append(cell_info)
            for nid in node_indices:
                if nid not in self.node_to_cells: self.node_to_cells[nid] = []
                self.node_to_cells[nid].append(i)

    def solve_steady(self):
        # Load Power from config path
        ptrace_name = self.config.get('ptrace_file_path')
        power_densities = {}
        if ptrace_name:
            ptrace_path = os.path.join(self.base_dir, ptrace_name)
            with open(ptrace_path, 'r') as f:
                lines = f.readlines()
                header = lines[0].strip().split()
                data = [[float(x) for x in l.strip().split()] for l in lines[1:] if l.strip()]
                avg_p = np.mean(data, axis=0)
                p_map = dict(zip(header, avg_p))
                for u in self.config.get('power_units', []):
                    vol = u['dx'] * u['dy'] * u['dz']
                    power_densities[u['name']] = p_map.get(u['name'], 0.0) / vol if vol > 0 else 0.0

        n = len(self.cells)
        rows, cols, data, b = [], [], [], np.zeros(n)
        print(f"Assembling matrix for {n} cells...")
        for i, cell in enumerate(self.cells):
            # Conduction
            candidates = set()
            for nid in cell['nodes']: candidates.update(self.node_to_cells[nid])
            for j in candidates:
                if i >= j: continue
                neighbor = self.cells[j]
                shared = cell['nodes'].intersection(neighbor['nodes'])
                if len(shared) >= 4:
                    dist = neighbor['center'] - cell['center']; axis = np.argmax(np.abs(dist))
                    area = cell['dims'][(axis+1)%3] * cell['dims'][(axis+2)%3]
                    G = 1.0 / ((cell['dims'][axis]/2)/(cell['k']*area) + (neighbor['dims'][axis]/2)/(neighbor['k']*area))
                    rows.extend([i, j, i, j]); cols.extend([i, j, j, i]); data.extend([-G, -G, G, G])
            # Power
            for u in self.config.get('power_units', []):
                u_box = [u['lx'], u['ly'], u['lz'], u['lx']+u['dx'], u['ly']+u['dy'], u['lz']+u['dz']]
                v_inter = get_intersection_volume(cell['box'], u_box)
                if v_inter > 0: b[i] -= v_inter * power_densities.get(u['name'], 0.0)
            
        # BCs
        for bc in self.config.get('boundary_conditions', []):
            if bc.get('type') == 'convection':
                h, T_inf, sel = bc.get('h', 0.0), bc.get('T_inf', 293.15), bc.get('selection', [])
                for i in [idx for idx, c in enumerate(self.cells) if c['tag'] in sel]:
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

    def save_results(self, T):
        # Fixed filenames as requested
        vtu_path = os.path.join(self.base_dir, "result.vtu")
        steady_path = os.path.join(self.base_dir, "units.steady")
        
        self.mesh.cell_data['Temperature'] = [np.array(T)]
        self.mesh.write(vtu_path)
        print(f"3D result saved to {vtu_path}")
        
        unit_temps = {u['name']: [] for u in self.config.get('power_units', [])}
        for i, cell in enumerate(self.cells):
            for u in self.config.get('power_units', []):
                u_box = [u['lx'], u['ly'], u['lz'], u['lx']+u['dx'], u['ly']+u['dy'], u['lz']+u['dz']]
                if get_intersection_volume(cell['box'], u_box) > 0.5 * np.prod(cell['dims']):
                    unit_temps[u['name']].append(T[i])
        
        with open(steady_path, 'w') as f:
            for name, ts in unit_temps.items():
                f.write(f"{name}\t{np.mean(ts) if ts else 0.0:.2f}\n")
        print(f"Unit results saved to {steady_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2: print("Usage: python solver.py <config.toml>")
    else:
        s = FVMSolver(sys.argv[1]); T = s.solve_steady(); s.save_results(T)
