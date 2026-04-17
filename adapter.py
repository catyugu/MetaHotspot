import toml
import os
import re
import gmsh
import numpy as np

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
                if not line or line.startswith('#') or not line.startswith('-'): continue
                parts = re.split(r'\s+', line, 1)
                key = parts[0][1:]
                value = parts[1] if len(parts) > 1 else True
                try: value = float(value)
                except ValueError: pass
                config[key] = value
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
                layer_num = int(lines[i])
                lateral, power = lines[i+1].upper() == 'Y', lines[i+2].upper() == 'Y'
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
        # all_entities is a list of {'name', 'lx', 'ly', 'lz', 'dx', 'dy', 'dz', 'tag'}
        xs = set(); ys = set(); zs = set()
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
        unit_to_entity = {}
        for i, u in enumerate(all_entities):
            tag = gmsh.model.addDiscreteEntity(3)
            gmsh.model.setPhysicalName(3, tag, u['name'])
            gmsh.model.addPhysicalGroup(3, [tag], u['tag'])
            unit_to_entity[i] = tag
        
        node_id = 1; node_map = {}; all_t = []; all_c = []
        for k, z in enumerate(zs):
            for j, y in enumerate(ys):
                for i, x in enumerate(xs):
                    all_t.append(node_id); all_c.extend([x,y,z]); node_map[(i,j,k)] = node_id; node_id += 1
        gmsh.model.mesh.addNodes(3, list(unit_to_entity.values())[0], all_t, all_c)

        elem_id = 1; ent_elems = {t: [] for t in unit_to_entity.values()}
        for k in range(len(zs)-1):
            for j in range(len(ys)-1):
                for i in range(len(xs)-1):
                    cx, cy, cz = (xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2, (zs[k]+zs[k+1])/2
                    found = -1
                    for e_idx, u in enumerate(all_entities):
                        if (u['lx']-1e-9<=cx<=u['lx']+u['dx']+1e-9 and u['ly']-1e-9<=cy<=u['ly']+u['dy']+1e-9 and u['lz']-1e-9<=cz<=u['lz']+u['dz']+1e-9):
                            # Prioritize smaller entities (like chips) over larger ones (like sink) for cell assignment
                            if found == -1 or (all_entities[e_idx]['dx'] * all_entities[e_idx]['dy'] < all_entities[found]['dx'] * all_entities[found]['dy']):
                                found = e_idx
                    if found != -1:
                        nodes = [node_map[(i,j,k)], node_map[(i+1,j,k)], node_map[(i+1,j+1,k)], node_map[(i,j+1,k)],
                                 node_map[(i,j,k+1)], node_map[(i+1,j,k+1)], node_map[(i+1,j+1,k+1)], node_map[(i,j+1,k+1)]]
                        ent_elems[unit_to_entity[found]].append((elem_id, nodes)); elem_id += 1
        
        for t, elems in ent_elems.items():
            if elems:
                e_ids = [e[0] for e in elems]; n_ids = []
                for e in elems: n_ids.extend(e[1])
                gmsh.model.mesh.addElements(3, t, [5], [e_ids], [n_ids])
        gmsh.write(output_path); gmsh.finalize()

def convert_hotspot_to_metahotspot(example_dir, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    parser = HotSpotParser(); config = parser.parse_config(os.path.join(example_dir, 'example.config'))
    materials = parser.parse_materials(os.path.join(example_dir, 'example.materials'))
    if 'silicon' not in materials: materials['silicon'] = {'k': 130.0, 'cp': 1.63e6, 'fluid': False}
    if 'copper' not in materials: materials['copper'] = {'k': 400.0, 'cp': 3.44e6, 'fluid': False}
    if 'aluminum' not in materials: materials['aluminum'] = {'k': 237.0, 'cp': 2.42e6, 'fluid': False}
    
    total_w, total_h = 0.0, 0.0
    for root, _, files in os.walk(example_dir):
        for f in files:
            if f.endswith('.flp'):
                for u in parser.parse_flp(os.path.join(root, f)):
                    total_w = max(total_w, u['left_x'] + u['width']); total_h = max(total_h, u['bottom_y'] + u['height'])
    if total_w == 0: total_w, total_h = 0.01, 0.01

    all_entities = [] # Used for Mesher
    power_units = []  # Used for heat sources in TOML
    domain_assignment = {}
    z_cursor = 0.0
    tag_counter = 1
    
    lcf_path = next((os.path.join(example_dir, f) for f in os.listdir(example_dir) if f.endswith('.lcf')), None)
    if lcf_path:
        for layer in parser.parse_lcf(lcf_path):
            mat = f"layer_{layer['id']}_mat" if layer['type']=='numeric' else layer['material']
            if layer['type']=='numeric': materials[mat] = {'k': layer['k'], 'cp': layer['cp'], 'fluid': False}
            if mat not in domain_assignment: domain_assignment[mat] = []
            
            flp_name = layer['flp_file']; flp_path = os.path.join(example_dir, flp_name)
            for u in parser.parse_flp(flp_path):
                entity = {'name':u['name'], 'lx':u['left_x'], 'ly':u['bottom_y'], 'lz':z_cursor, 'dx':u['width'], 'dy':u['height'], 'dz':layer['thickness'], 'tag': tag_counter, 'layer_id': layer['id']}
                all_entities.append(entity)
                if layer['power']:
                    power_units.append({'name':u['name'], 'lx':u['left_x'], 'ly':u['bottom_y'], 'lz':z_cursor, 'dx':u['width'], 'dy':u['height'], 'dz':layer['thickness'], 'material':mat, 'layer_id':layer['id'], 'domain_id': tag_counter})
                domain_assignment[mat].append(tag_counter)
                tag_counter += 1
            z_cursor += layer['thickness']
    else:
        flp_path = next((os.path.join(example_dir, f) for f in os.listdir(example_dir) if f.endswith('.flp')), None)
        if flp_path:
            t_chip = config.get('t_chip', 0.00015)
            if 'silicon' not in domain_assignment: domain_assignment['silicon'] = []
            for u in parser.parse_flp(flp_path):
                entity = {'name':u['name'], 'lx':u['left_x'], 'ly':u['bottom_y'], 'lz':z_cursor, 'dx':u['width'], 'dy':u['height'], 'dz':t_chip, 'tag': tag_counter, 'layer_id': 0}
                all_entities.append(entity)
                power_units.append({'name':u['name'], 'lx':u['left_x'], 'ly':u['bottom_y'], 'lz':z_cursor, 'dx':u['width'], 'dy':u['height'], 'dz':t_chip, 'material':'silicon', 'layer_id':0, 'domain_id': tag_counter})
                domain_assignment['silicon'].append(tag_counter)
                tag_counter += 1
            z_cursor += t_chip

    # Structural Domains (Passive)
    t_tim, t_spreader, t_sink = config.get('t_tim', 0.00002), config.get('t_spreader', 0.001), config.get('t_sink', 0.0069)
    s_spreader, s_sink = config.get('s_spreader', 0.03), config.get('s_sink', 0.06)

    def add_structural_domain(name, thickness, side, material, tag):
        nonlocal z_cursor
        lx, ly = (total_w - side) / 2, (total_h - side) / 2
        all_entities.append({'name': name, 'lx': lx, 'ly': ly, 'lz': z_cursor, 'dx': side, 'dy': side, 'dz': thickness, 'tag': tag, 'layer_id': tag})
        if material not in domain_assignment: domain_assignment[material] = []
        domain_assignment[material].append(tag)
        z_cursor += thickness

    add_structural_domain("TIM", t_tim, total_w, "silicon", 1000)
    add_structural_domain("Spreader", t_spreader, s_spreader, "copper", 1001)
    add_structural_domain("Sink", t_sink, s_sink, "aluminum", 1002)

    top_selection = [1002] # Convection applied to Sink (tag 1002)
    toml_data = {
        'simulation_type': 'steady', 'materials': materials, 'domain_material_assignment': domain_assignment,
        'power_units': power_units,
        'boundary_conditions': [{'name': 'sink_convection', 'type': 'convection', 'h': 1.0 / (config.get('r_convec', 0.1) * (s_sink**2)), 'T_inf': config.get('ambient', 293.15), 'selection': top_selection}]
    }
    with open(os.path.join(output_dir, 'solver_config.toml'), 'w') as f: toml.dump(toml_data, f)
    Mesher(mesh_size=0.002).generate_mesh_robust(all_entities, os.path.join(output_dir, 'mesh.msh'))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3: print("Usage: python adapter.py <input_dir> <output_dir>")
    else: convert_hotspot_to_metahotspot(sys.argv[1], sys.argv[2])
