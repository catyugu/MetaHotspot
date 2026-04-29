import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metahotspot.converter import convert_hotspot_with_modes
from metahotspot.gmsh_mesher import GmshMesher


def _convert_and_mesh_batch(
    hotspot_examples_dir: str, output_root: str, mode: str
) -> None:
    """Convert and mesh HotSpot examples example1~example4."""
    for name in ("example1", "example2", "example3", "example4"):
        in_dir = os.path.join(hotspot_examples_dir, name)
        out_dir = os.path.join(output_root, name)
        created = convert_hotspot_with_modes(in_dir, out_dir, mode=mode)
        print(f"[CONVERT] {name} -> {out_dir}")
        for config_path in created:
            print(f"[CONVERT]   wrote {config_path}")

        # Mesh each generated config
        base_dir = out_dir
        if mode == "both":
            configs_to_mesh = [os.path.join(base_dir, "solver_config_steady.toml")]
        else:
            configs_to_mesh = created

        for config_path in configs_to_mesh:
            if not os.path.exists(config_path):
                continue
            mesher = GmshMesher()
            mesher.generate_mesh(config_path)
            mesh_path = os.path.join(base_dir, "mesh.msh")
            mesher.finalize(mesh_path)
            print(f"[MESH] {name} -> {mesh_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HotSpot example inputs to MetaHotspot solver config and mesh."
    )
    parser.add_argument("input_dir", nargs="?", help="Hotspot example directory")
    parser.add_argument("output_dir", nargs="?", help="Output directory")
    parser.add_argument(
        "--mode",
        choices=("steady", "transient", "both"),
        default="both",
        help="Generate steady, transient, or both solver configs (default: both)",
    )
    parser.add_argument(
        "--batch-four",
        action="store_true",
        help="Convert and mesh HotSpot examples example1~example4 in one command",
    )
    parser.add_argument(
        "--hotspot-examples-dir",
        default="Hotspot/examples",
        help="Path to Hotspot examples root (default: Hotspot/examples)",
    )
    parser.add_argument(
        "--output-root",
        default="examples/hotspot_converted",
        help="Batch output root (default: examples/hotspot_converted)",
    )
    args = parser.parse_args()

    if args.batch_four:
        _convert_and_mesh_batch(args.hotspot_examples_dir, args.output_root, args.mode)
        return

    if not args.input_dir or not args.output_dir:
        parser.error(
            "input_dir and output_dir are required unless --batch-four is used"
        )

    # Single example: convert then mesh
    created = convert_hotspot_with_modes(
        args.input_dir, args.output_dir, mode=args.mode
    )
    for config_path in created:
        print(f"[CONVERT] wrote {config_path}")

    # Mesh the steady config (if mode=both, mesh steady only for batch efficiency)
    base_dir = args.output_dir
    config_to_mesh = os.path.join(base_dir, "solver_config_steady.toml")
    if os.path.exists(config_to_mesh):
        mesher = GmshMesher()
        mesher.generate_mesh(config_to_mesh)
        mesh_path = os.path.join(base_dir, "mesh.msh")
        mesher.finalize(mesh_path)
        print(f"[MESH] wrote {mesh_path}")


if __name__ == "__main__":
    main()
