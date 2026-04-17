import toml
import os
import re
import gmsh
import numpy as np
import shutil

class HotSpotParser:
    @staticmethod
    def parse_flp(file_path):
        units = []
        if not os.path.exists(file_path): return units
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                parts = re.split(r'\s+', line)
                if len(parts) >= 5:
                    units.append({
                        'name': parts[0], 'width': float(parts[1]), 'height': float(parts[2]),
                        'left_x': float(parts[3]), 'bottom_y': float(parts[4])
                    })
        return units

    @staticmethod
    def parse_config(file_path):
        config = {}
        if not os.path.exists(file_path): return config
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                # Match both "-key value" and "-key=value"
                match = re.match(r'^-(\w+)\s+([^#]+)', line)
                if match:
                    key, val = match.groups()
                    val = val.strip()
                    try: config[key] = float(val)
                    except: config[key] = val
        return config

    @staticmethod
    def parse_materials(file_path):
        materials = {}
        if not os.path.exists(file_path): return materials
        with open(file_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
            i = 0
            while i < len(lines):
                name = lines[i]; m_type = lines[i+1]; k = float(lines[i+2]); cp = float(lines[i+3])
                if m_type.lower() == 'fluid':
                    v = float(lines[i+4]); materials[name] = {'k': k, 'cp': cp, 'fluid': True, 'dynamic_viscosity': v}; i += 5
                else:
                    materials[name] = {'k': k, 'cp': cp, 'fluid': False}; i += 4
        return materials

    @staticmethod
    def parse_lcf(file_path):
        layers = []
        if not os.path.exists(file_path): return layers
        with open(file_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
            i = 0
            while i < len(lines):
                layer_num = int(lines[i]); lateral, power = lines[i+1].upper() == 'Y', lines[i+2].upper() == 'Y'
                val_3 = lines[i+3]
                try:
                    cp = float(val_3); res = float(lines[i+4]); thick = float(lines[i+5]); flp = lines[i+6]
                    layers.append({'id': layer_num, 'power': power, 'cp': cp, 'k': 1.0/res if res!=0 else 0, 'thickness': thick, 'flp_file': flp, 'type': 'numeric'})
                    i += 7
                except ValueError:
                    thick = float(lines[i+4]); flp = lines[i+5]
                    layers.append({'id': layer_num, 'power': power, 'material': val_3, 'thickness': thick, 'flp_file': flp, 'type': 'named'})
                    i += 6
        return layers

class Mesher:
    def __init__(self, mesh_size=0.001):
        self.mesh_size = mesh_size
        gmsh.initialize()
        gmsh.model.add("MetaHotspotMesh")

    def generate_mesh_robust(self, all_entities, output_path):
        xs, ys, zs = set(), set(), set()
        for u in all_entities:
            xs.add(u['lx']); xs.add(u['lx']+u['dx']); ys.add(u['ly']); ys.add(u['ly']+u['dy']); zs.add(u['lz']); zs.add(u['lz']+u['dz'])
        
        def subdivide(coords, target):
            sorted_c = sorted(list(coords)); unique = [sorted_c[0]]
            for c in sorted_c[1:]:
                if c - unique[-1] > 1e-12: unique.append(c)
            res = []
            for i in range(len(unique)-1):
                c1, c2 = unique[i], unique[i+1]; res.append(c1)
                if c2-c1 > target*1.1:
                    n = max(1, int(round((c2-c1)/target)))
                    for j in range(1, n): res.append(c1 + j*(c2-c1)/n)
            res.append(unique[-1]); return res

        xs = subdivide(xs, self.mesh_size); ys = subdivide(ys, self.mesh_size); zs = subdivide(zs, self.mesh_size)
        layer_tags = sorted(list(set(u['tag'] for u in all_entities)))
        tag_to_ent = {}
        for t in layer_tags:
            d_tag = gmsh.model.addDiscreteEntity(3); gmsh.model.addPhysicalGroup(3, [d_tag], t); tag_to_ent[t] = d_tag
        
        node_id = 1; node_map = {}; all_t, all_c = [], []
        for k, z in enumerate(zs):
            for j, y in enumerate(ys):
                for i, x in enumerate(xs):
                    all_t.append(node_id); all_c.extend([x,y,z]); node_map[(i,j,k)] = node_id; node_id += 1
        gmsh.model.mesh.addNodes(3, list(tag_to_ent.values())[0], all_t, all_c)

        elem_id = 1; ent_elems = {t: [] for t in tag_to_ent.values()}
        for k in range(len(zs)-1):
            for j in range(len(ys)-1):
                for i in range(len(xs)-1):
                    cx, cy, cz = (xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2, (zs[k]+zs[k+1])/2
                    found_tag = -1; best_area = float('inf')
                    for u in all_entities:
                        if (u['lx']-1e-9<=cx<=u['lx']+u['dx']+1e-9 and u['ly']-1e-9<=cy<=u['ly']+u['dy']+1e-9 and u['lz']-1e-9<=cz<=u['lz']+u['dz']+1e-9):
                            if u['dx']*u['dy'] < best_area: found_tag = u['tag']; best_area = u['dx']*u['dy']
                    if found_tag != -1:
                        nodes = [node_map[(i,j,k)], node_map[(i+1,j,k)], node_map[(i+1,j+1,k)], node_map[(i,j+1,k)],
                                 node_map[(i,j,k+1)], node_map[(i+1,j,k+1)], node_map[(i+1,j+1,k+1)], node_map[(i,j+1,k+1)]]
                        ent_elems[tag_to_ent[found_tag]].append((elem_id, nodes)); elem_id += 1
        
        for t_ent, elems in ent_elems.items():
            if elems:
                e_ids = [e[0] for e in elems]; n_ids = []
                for e in elems: n_ids.extend(e[1])
                gmsh.model.mesh.addElements(3, t_ent, [5], [e_ids], [n_ids])
        gmsh.write(output_path); gmsh.finalize()

def convert_hotspot_to_metahotspot(example_dir, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    parser = HotSpotParser(); config = parser.parse_config(os.path.join(example_dir, 'example.config'))
    materials = parser.parse_materials(os.path.join(example_dir, 'example.materials'))
    
    # Material Library Completeness with realistic HotSpot defaults
    std_mats = {
        'silicon': {'k': 130.0, 'cp': 1.63e6, 'fluid': False},
        'copper': {'k': 400.0, 'cp': 3.44e6, 'fluid': False},
        'aluminum': {'k': 237.0, 'cp': 2.42e6, 'fluid': False},
        'tim': {'k': 4.0, 'cp': 4.0e6, 'fluid': False} # Realistic TIM k=1/0.25
    }
    for m_name, props in std_mats.items():
        if m_name not in materials: materials[m_name] = props

    total_w, total_h = 0.0, 0.0
    for root, _, files in os.walk(example_dir):
        for f in files:
            if f.endswith('.flp'):
                for u in parser.parse_flp(os.path.join(root, f)):
                    total_w = max(total_w, u['left_x'] + u['width']); total_h = max(total_h, u['bottom_y'] + u['height'])
    if total_w == 0: total_w, total_h = 0.01, 0.01

    ptrace_src = next((os.path.join(example_dir, f) for f in os.listdir(example_dir) if f.endswith('.ptrace')), None)
    ptrace_local_path = ""
    if ptrace_src:
        ptrace_name = os.path.basename(ptrace_src)
        shutil.copy(ptrace_src, os.path.join(output_dir, ptrace_name))
        ptrace_local_path = ptrace_name

    all_entities, power_units, domain_assignment, z_cursor = [], [], {}, 0.0
    lcf_path = next((os.path.join(example_dir, f) for f in os.listdir(example_dir) if f.endswith('.lcf')), None)
    
    if lcf_path:
        for layer in parser.parse_lcf(lcf_path):
            tag = layer['id'] + 1
            mat = f"layer_{layer['id']}_mat" if layer['type']=='numeric' else layer['material']
            if layer['type']=='numeric': materials[mat] = {'k': layer['k'], 'cp': layer['cp'], 'fluid': False}
            if mat not in domain_assignment: domain_assignment[mat] = []
            domain_assignment[mat].append(tag)
            for u in parser.parse_flp(os.path.join(example_dir, layer['flp_file'])):
                all_entities.append({'lx':u['left_x'], 'ly':u['bottom_y'], 'lz':z_cursor, 'dx':u['width'], 'dy':u['height'], 'dz':layer['thickness'], 'tag': tag})
                if layer['power']: power_units.append({'name':u['name'], 'lx':u['left_x'], 'ly':u['bottom_y'], 'lz':z_cursor, 'dx':u['width'], 'dy':u['height'], 'dz':layer['thickness']})
            z_cursor += layer['thickness']
    else:
        flp_path = next((os.path.join(example_dir, f) for f in os.listdir(example_dir) if f.endswith('.flp')), None)
        if flp_path:
            tag, t_chip = 1, config.get('t_chip', 0.00015)
            if 'silicon' not in domain_assignment: domain_assignment['silicon'] = []
            domain_assignment['silicon'].append(tag)
            for u in parser.parse_flp(flp_path):
                all_entities.append({'lx':u['left_x'], 'ly':u['bottom_y'], 'lz':z_cursor, 'dx':u['width'], 'dy':u['height'], 'dz':t_chip, 'tag': tag})
                power_units.append({'name':u['name'], 'lx':u['left_x'], 'ly':u['bottom_y'], 'lz':z_cursor, 'dx':u['width'], 'dy':u['height'], 'dz':t_chip})
            z_cursor += t_chip

    def add_pkg(name, thick, side, mat, tag):
        nonlocal z_cursor; lx, ly = (total_w-side)/2, (total_h-side)/2
        all_entities.append({'lx':lx, 'ly':ly, 'lz':z_cursor, 'dx':side, 'dy':side, 'dz':thick, 'tag':tag})
        if mat not in domain_assignment: domain_assignment[mat] = []
        domain_assignment[mat].append(tag); z_cursor += thick

    # TIM layer uses "tim" material (k=4) instead of silicon
    add_pkg("TIM", config.get('t_tim', 0.00002), total_w, "tim", 1000)
    add_pkg("Spreader", config.get('t_spreader', 0.001), config.get('s_spreader', 0.03), "copper", 1001)
    add_pkg("Sink", config.get('t_sink', 0.0069), config.get('s_sink', 0.06), "aluminum", 1002)

    toml_data = {
        'simulation_type': 'steady', 'materials': materials, 'domain_material_assignment': domain_assignment,
        'mesh_file_path': 'mesh.msh', 'ptrace_file_path': ptrace_local_path,
        'power_units': power_units,
        'ambient': config.get('ambient', 318.15), # Default HotSpot ambient 45C
        'boundary_conditions': [{'name': 'sink_conv', 'type': 'convection', 'h': 1.0/(config.get('r_convec',0.1)*(config.get('s_sink',0.06)**2)), 'T_inf': config.get('ambient',318.15), 'selection': [1002]}]
    }
    with open(os.path.join(output_dir, 'solver_config.toml'), 'w') as f: toml.dump(toml_data, f)
    # Reasonable mesh size for Python solver
    Mesher(mesh_size=0.002).generate_mesh_robust(all_entities, os.path.join(output_dir, 'mesh.msh'))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3: print("Usage: python adapter.py <input_dir> <output_dir>")
    else: convert_hotspot_to_metahotspot(sys.argv[1], sys.argv[2])
