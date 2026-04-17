import toml
import os
import re
import gmsh

class HotSpotParser:
    @staticmethod
    def parse_flp(file_path):
        units = []
        if not os.path.exists(file_path):
            return units
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = re.split(r'\s+', line)
                if len(parts) >= 5:
                    units.append({
                        'name': parts[0],
                        'width': float(parts[1]),
                        'height': float(parts[2]),
                        'left_x': float(parts[3]),
                        'bottom_y': float(parts[4])
                    })
        return units

    @staticmethod
    def parse_config(file_path):
        config = {}
        if not os.path.exists(file_path):
            return config
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('-'):
                    parts = re.split(r'\s+', line, 1)
                    key = parts[0][1:]
                    value = parts[1] if len(parts) > 1 else True
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    config[key] = value
        return config

    @staticmethod
    def parse_materials(file_path):
        materials = {}
        if not os.path.exists(file_path):
            return materials
        with open(file_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
            i = 0
            while i < len(lines):
                name = lines[i]
                m_type = lines[i+1]
                k = float(lines[i+2])
                cp = float(lines[i+3])
                if m_type.lower() == 'fluid':
                    viscosity = float(lines[i+4])
                    materials[name] = {'k': k, 'cp': cp, 'fluid': True, 'dynamic_viscosity': viscosity}
                    i += 5
                else:
                    materials[name] = {'k': k, 'cp': cp, 'fluid': False}
                    i += 4
        return materials

    @staticmethod
    def parse_lcf(file_path):
        layers = []
        if not os.path.exists(file_path):
            return layers
        with open(file_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
            i = 0
            while i < len(lines):
                layer_num = int(lines[i])
                lateral = lines[i+1].upper() == 'Y'
                power = lines[i+2].upper() == 'Y'
                
                val_3 = lines[i+3]
                try:
                    cp = float(val_3)
                    resistivity = float(lines[i+4])
                    thickness = float(lines[i+5])
                    flp_file = lines[i+6]
                    layers.append({
                        'id': layer_num, 'lateral': lateral, 'power': power,
                        'cp': cp, 'k': 1.0/resistivity if resistivity != 0 else 0,
                        'thickness': thickness, 'flp_file': flp_file, 'type': 'numeric'
                    })
                    i += 7
                except ValueError:
                    mat_name = val_3
                    thickness = float(lines[i+4])
                    flp_file = lines[i+5]
                    layers.append({
                        'id': layer_num, 'lateral': lateral, 'power': power,
                        'material': mat_name, 'thickness': thickness, 'flp_file': flp_file, 'type': 'named'
                    })
                    i += 6
        return layers

class Mesher:
    def __init__(self, mesh_size=0.001):
        self.mesh_size = mesh_size
        gmsh.initialize()
        gmsh.model.add("MetaHotspotMesh")

    def generate_mesh_robust(self, power_units, output_path):
        xs = set(); ys = set(); zs = set()
        for unit in power_units:
            xs.add(unit['lx']); xs.add(unit['lx'] + unit['dx'])
            ys.add(unit['ly']); ys.add(unit['ly'] + unit['dy'])
            zs.add(unit['lz']); zs.add(unit['lz'] + unit['dz'])
        
        def subdivide(coords, target_size):
            sorted_coords = sorted(list(coords))
            unique = []
            if sorted_coords:
                unique.append(sorted_coords[0])
                for c in sorted_coords[1:]:
                    if c - unique[-1] > 1e-12: unique.append(c)
            new_c = []
            for i in range(len(unique) - 1):
                c1, c2 = unique[i], unique[i+1]
                new_c.append(c1)
                dist = c2 - c1
                if dist > target_size * 1.1:
                    n = max(1, int(round(dist / target_size)))
                    for j in range(1, n): new_c.append(c1 + j * (dist / n))
            if unique: new_c.append(unique[-1])
            return new_c

        xs = subdivide(xs, self.mesh_size)
        ys = subdivide(ys, self.mesh_size)
        zs = subdivide(zs, self.mesh_size)

        unit_to_entity = {}
        for i, unit in enumerate(power_units):
            tag = gmsh.model.addDiscreteEntity(3)
            gmsh.model.setPhysicalName(3, tag, unit['name'])
            gmsh.model.addPhysicalGroup(3, [tag], i + 1)
            unit_to_entity[i] = tag

        if not unit_to_entity: return
        primary_entity = unit_to_entity[0]

        node_id = 1; node_map = {}; all_tags = []; all_coords = []
        for k, z in enumerate(zs):
            for j, y in enumerate(ys):
                for i, x in enumerate(xs):
                    all_tags.append(node_id); all_coords.extend([x, y, z])
                    node_map[(i, j, k)] = node_id; node_id += 1
        gmsh.model.mesh.addNodes(3, primary_entity, all_tags, all_coords)
        
        elem_id = 1; entity_elems = {tag: [] for tag in unit_to_entity.values()}
        for k in range(len(zs) - 1):
            for j in range(len(ys) - 1):
                for i in range(len(xs) - 1):
                    cx = (xs[i] + xs[i+1]) / 2; cy = (ys[j] + ys[j+1]) / 2; cz = (zs[k] + zs[k+1]) / 2
                    found_idx = -1
                    for u_idx, unit in enumerate(power_units):
                        if (unit['lx']-1e-9 <= cx <= unit['lx']+unit['dx']+1e-9 and
                            unit['ly']-1e-9 <= cy <= unit['ly']+unit['dy']+1e-9 and
                            unit['lz']-1e-9 <= cz <= unit['lz']+unit['dz']+1e-9):
                            found_idx = u_idx; break
                    if found_idx != -1:
                        nodes = [node_map[(i,j,k)], node_map[(i+1,j,k)], node_map[(i+1,j+1,k)], node_map[(i,j+1,k)],
                                 node_map[(i,j,k+1)], node_map[(i+1,j,k+1)], node_map[(i+1,j+1,k+1)], node_map[(i,j+1,k+1)]]
                        entity_elems[unit_to_entity[found_idx]].append((elem_id, nodes)); elem_id += 1
        
        for tag, elems in entity_elems.items():
            if elems:
                e_ids = [e[0] for e in elems]; n_ids = []
                for e in elems: n_ids.extend(e[1])
                gmsh.model.mesh.addElements(3, tag, [5], [e_ids], [n_ids])
        gmsh.write(output_path)
        gmsh.finalize()

def convert_hotspot_to_metahotspot(example_dir, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    parser = HotSpotParser()
    config = parser.parse_config(os.path.join(example_dir, 'example.config'))
    materials = parser.parse_materials(os.path.join(example_dir, 'example.materials'))
    
    # Standard materials
    if 'silicon' not in materials: materials['silicon'] = {'k': 130.0, 'cp': 1.63e6, 'fluid': False}
    
    # Chip dimensions
    total_w, total_h = 0.0, 0.0
    for root, _, files in os.walk(example_dir):
        for f in files:
            if f.endswith('.flp'):
                for u in parser.parse_flp(os.path.join(root, f)):
                    total_w = max(total_w, u['left_x'] + u['width'])
                    total_h = max(total_h, u['bottom_y'] + u['height'])
    if total_w == 0: total_w, total_h = 0.01, 0.01

    lcf_path = next((os.path.join(example_dir, f) for f in os.listdir(example_dir) if f.endswith('.lcf')), None)
    
    power_units = []
    if lcf_path:
        layers = parser.parse_lcf(lcf_path)
        z_offset = 0.0
        for layer in layers:
            # ... (Material logic)
            if layer['type'] == 'numeric':
                mat_name = f"layer_{layer['id']}_mat"
                materials[mat_name] = {'k': layer['k'], 'cp': layer['cp'], 'fluid': False}
            else:
                mat_name = layer['material']
            
            flp_name = layer['flp_file']
            flp_path = os.path.join(example_dir, flp_name)
            
            if flp_name.endswith('.csv'):
                units = parser.parse_csv_layer(flp_path, total_w, total_h, layer['thickness'], z_offset)
                for u in units:
                    u['material'] = 'water' if u['code'] in [1, 2, 3] else mat_name
                    u['layer_id'] = layer['id']
                    u['dx'] = u['dx']; u['dy'] = u['dy'] # Already set
                    power_units.append(u)
            else:
                units = parser.parse_flp(flp_path)
                for u in units:
                    u.update({'lx': u['left_x'], 'ly': u['bottom_y'], 'lz': z_offset, 
                              'dx': u['width'], 'dy': u['height'], 'dz': layer['thickness'],
                              'material': mat_name, 'layer_id': layer['id']})
                    power_units.append(u)
            z_offset += layer['thickness']
    else:
        # SINGLE LAYER FALLBACK
        flp_path = next((os.path.join(example_dir, f) for f in os.listdir(example_dir) if f.endswith('.flp')), None)
        if flp_path:
            t_chip = config.get('t_chip', 0.00015)
            units = parser.parse_flp(flp_path)
            for u in units:
                u.update({'lx': u['left_x'], 'ly': u['bottom_y'], 'lz': 0.0, 
                          'dx': u['width'], 'dy': u['height'], 'dz': t_chip,
                          'material': 'silicon', 'layer_id': 0})
                power_units.append(u)
    # 4. Generate Boundary Conditions
    # Calculate global top surface area for h_conv conversion
    z_max = max(u['lz'] + u['dz'] for u in power_units)
    z_min = min(u['lz'] for u in power_units)

    top_selection = []
    bottom_selection = []
    total_top_area = 0.0

    for i, unit in enumerate(power_units):
        # Using a small epsilon for float comparison
        if abs((unit['lz'] + unit['dz']) - z_max) < 1e-9:
            top_selection.append(i + 1)
            total_top_area += unit['dx'] * unit['dy']
        if abs(unit['lz'] - z_min) < 1e-9:
            bottom_selection.append(i + 1)

    r_convec = config.get('r_convec', 0.1)
    ambient = config.get('ambient', 293.15)

    # h = 1 / (R * Area)
    h_top = 1.0 / (r_convec * total_top_area) if total_top_area > 0 else 0

    toml_data = {
        'simulation_type': 'steady',
        'materials': materials,
        'domain_material_assignment': {},
        'power_units': power_units,
        'boundary_conditions': [
            {
                'name': 'top_convection',
                'type': 'convection',
                'h': h_top,
                'T_inf': ambient,
                'selection': top_selection
            }
        ]
    }

    # If a secondary convection is defined in HotSpot (r_convec_bot), we'd add it here.
    
    for i, unit in enumerate(power_units):
        mat = unit['material']
        if mat not in toml_data['domain_material_assignment']: toml_data['domain_material_assignment'][mat] = []
        toml_data['domain_material_assignment'][mat].append(i + 1)

    with open(os.path.join(output_dir, 'solver_config.toml'), 'w') as f: toml.dump(toml_data, f)
    mesher = Mesher(mesh_size=0.0005)
    mesher.generate_mesh_robust(power_units, os.path.join(output_dir, 'mesh.msh'))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3: print("Usage: python adapter.py <input_dir> <output_dir>")
    else: convert_hotspot_to_metahotspot(sys.argv[1], sys.argv[2])
