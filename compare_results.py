"""Compare computed temperatures against reference values in original XML files."""

import sys
import xml.etree.ElementTree as ET
import math

NS = {
    'dc': 'http://schemas.datacontract.org/2004/07/ThermalSim.Models',
    'a': 'http://schemas.microsoft.com/2003/10/Serialization/Arrays',
    'i': 'http://www.w3.org/2001/XMLSchema-instance',
}

def find_element(parent, local_tag):
    """Find an element by local tag name, handling namespaces."""
    # Try with default namespace
    result = parent.find(f'dc:{local_tag}', NS)
    if result is not None:
        return result
    # Try without namespace
    result = parent.find(local_tag)
    if result is not None:
        return result
    # Search all children for matching local name
    for child in parent:
        tag_local = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if tag_local == local_tag:
            return child
    return None

def find_all_elements(parent, local_tag):
    """Find all elements by local tag name."""
    results = []
    for child in parent:
        tag_local = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if tag_local == local_tag:
            results.append(child)
    return results

def parse_double_list(data_elem):
    """Parse a list of doubles from a Data element."""
    values = []
    for child in data_elem:
        tag_local = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if tag_local != 'double':
            continue
        text = child.text.strip()
        if text == 'NaN' or text.lower() == 'nan':
            values.append(float('nan'))
        else:
            values.append(float(text))
    return values

def get_temps_from_xml(xml_path):
    """Extract temperature values and grid dimensions from the Results section."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    results = find_element(root, 'Results')
    if results is None:
        print(f"  No Results section in {xml_path}")
        return None

    any_type = find_element(results, 'anyType')
    if any_type is None:
        print(f"  No anyType in Results of {xml_path}")
        return None

    values_elem = find_element(any_type, 'Values')
    if values_elem is None:
        print(f"  No Values in {xml_path}")
        return None

    data_elem = find_element(values_elem, 'Data')
    if data_elem is None:
        print(f"  No Data in {xml_path}")
        return None

    size_x = int(find_element(values_elem, 'SizeX').text)
    size_y = int(find_element(values_elem, 'SizeY').text)
    size_z = int(find_element(values_elem, 'SizeZ').text)

    temps = parse_double_list(data_elem)
    return temps, (size_x, size_y, size_z), len(temps)

def compare_temps(ref_temps, comp_temps, ref_size, comp_size, tolerance=5.0):
    """Compare reference and computed temperatures within tolerance."""
    ref_total = ref_size[0] * ref_size[1] * ref_size[2]
    comp_total = comp_size[0] * comp_size[1] * comp_size[2]

    if ref_total != comp_total:
        print(f"  Size mismatch: ref={ref_size} ({ref_total}), comp={comp_size} ({comp_total})")
        return None

    max_error = 0.0
    total_error = 0.0
    num_comparable = 0
    num_exceed = 0

    for i in range(ref_total):
        ref_val = ref_temps[i]
        comp_val = comp_temps[i]

        if math.isnan(ref_val) and math.isnan(comp_val):
            continue
        if math.isnan(ref_val) or math.isnan(comp_val):
            continue

        error = abs(ref_val - comp_val)
        max_error = max(max_error, error)
        total_error += error
        num_comparable += 1

        if error > tolerance:
            num_exceed += 1
            if num_exceed <= 10:
                sx, sy, sz = ref_size
                x = i // (sy * sz)
                y = (i % (sy * sz)) // sz
                z = i % sz
                print(f"    Exceed: ({x},{y},{z}), ref={ref_val:.4f}, comp={comp_val:.4f}, err={error:.4f}")

    mean_error = total_error / num_comparable if num_comparable > 0 else 0.0
    return max_error, mean_error, num_comparable, num_exceed

def main():
    cases = [
        ("case1", "cases/original_steady_tests/case1.xml", "output_case1.xml"),
        ("case2", "cases/original_steady_tests/case2.xml", "output_case2.xml"),
        ("case3", "cases/original_steady_tests/case3.xml", "output_case3.xml"),
    ]

    tolerance = 5.0
    all_pass = True

    for name, ref_path, comp_path in cases:
        print(f"\n=== {name} ===")
        ref = get_temps_from_xml(ref_path)
        comp = get_temps_from_xml(comp_path)

        if ref is None or comp is None:
            all_pass = False
            continue

        ref_temps, ref_size, ref_count = ref
        comp_temps, comp_size, comp_count = comp

        print(f"  Reference: {ref_size} ({ref_count} values)")
        print(f"  Computed:  {comp_size} ({comp_count} values)")

        result = compare_temps(ref_temps, comp_temps, ref_size, comp_size, tolerance)
        if result is None:
            all_pass = False
            continue

        max_err, mean_err, num_comp, num_exceed = result
        print(f"  Comparable points: {num_comp}")
        print(f"  Max error: {max_err:.4f} K")
        print(f"  Mean error: {mean_err:.4f} K")
        print(f"  Points exceeding {tolerance}K: {num_exceed}")

        if max_err <= tolerance:
            print(f"  PASS")
        else:
            print(f"  FAIL")
            all_pass = False

    print(f"\n{'All cases PASSED!' if all_pass else 'Some cases FAILED!'}")
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())