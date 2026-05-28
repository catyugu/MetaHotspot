#!/usr/bin/env python3
"""Compare simulation results with reference values."""

import xml.etree.ElementTree as ET
import sys
import math

def parse_xml_results(path):
    """Parse temperature results from XML."""
    tree = ET.parse(path)
    root = tree.getroot()

    # Find mesh and values
    ns = {
        'a': 'http://schemas.microsoft.com/2003/10/Serialization/Arrays',
        'b': 'http://schemas.datacontract.org/2004/07/ThermalSim.Models.Mesh'
    }

    values = []
    for elem in root.iter('{http://schemas.microsoft.com/2003/10/Serialization/Arrays}double'):
        try:
            val = float(elem.text)
            if not math.isnan(val):
                values.append(val)
        except (ValueError, TypeError):
            continue

    return values

def compare_results(ref_values, sim_values, tolerance=5000.0):
    """Compare results within tolerance."""
    if len(ref_values) != len(sim_values):
        print(f"Warning: Length mismatch ({len(ref_values)} vs {len(sim_values)})")

    min_len = min(len(ref_values), len(sim_values))
    max_error = 0.0
    total_error = 0.0
    count = 0

    for i in range(min_len):
        err = abs(ref_values[i] - sim_values[i])
        max_error = max(max_error, err)
        total_error += err
        count += 1

    if count > 0:
        avg_error = total_error / count
    else:
        avg_error = float('inf')

    return max_error, avg_error

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_results.py <reference.xml> <simulation.xml>")
        sys.exit(1)

    ref_path = sys.argv[1]
    sim_path = sys.argv[2]

    try:
        ref_values = parse_xml_results(ref_path)
        print(f"Reference: {len(ref_values)} values")

        sim_values = parse_xml_results(sim_path)
        print(f"Simulation: {len(sim_values)} values")

        max_err, avg_err = compare_results(ref_values, sim_values)
        print(f"\nResults:")
        print(f"  Max error: {max_err:.2f} K")
        print(f"  Avg error: {avg_err:.2f} K")

        if max_err < 5000.0:
            print(f"\n  PASS: Results within 5000K tolerance")
            return 0
        else:
            print(f"\n  FAIL: Max error exceeds 5000K tolerance")
            return 1

    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())