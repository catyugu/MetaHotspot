"""Exercise the independent extractor and native reader in separate processes."""

import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    prepare, check = sys.argv[1:]
    folder = Path(__file__).resolve().parent
    xml = folder / "three_cells.xml"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(folder.parents[2] / "python"), env.get("PYTHONPATH")))
    )
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    extract = [
        sys.executable,
        "-m",
        "metahotspot.erom",
        str(xml),
        "--prepare",
        prepare,
        "--htc",
        "0",
        "1",
        "10",
        "--htc",
        "1",
        "2",
        "9",
        "--contour-degree",
        "0",
        "--tolerance",
        "1e-6",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        native = Path(tmp) / "test.mherom"

        rejected = subprocess.run(
            [*extract[:4], str(native), *extract[4:]],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected.returncode and "depends directly on HTC" in rejected.stderr, (
            rejected.stderr
        )
        assert not native.exists()

        subprocess.run(
            [*extract[:4], str(native), *extract[4:], "--probe", "off center"],
            env=env,
            check=True,
        )
        subprocess.run([check, str(native), str(xml)], check=True, env=env)

        original = native.read_bytes()
        rank, ns, no, nb, nf, _ = struct.unpack_from("<6I", original, 12)
        first_facet = 64 + 8 * (2 * rank * rank + rank * (ns + no + nb) + nb + 2 * rank)
        first_group = first_facet + 48 * nf
        first_group_name_size = struct.unpack_from("<I", original, first_group + 4)[0]
        first_facet_id = first_group + 16 + first_group_name_size + 32
        for offset, value, reason in (
            (8, struct.pack("<I", 99), "unsupported version"),
            (12, struct.pack("<I", 0xFFFFFFFF), "remaining file size"),
            (64, struct.pack("<d", float("nan")), "nonfinite matrix"),
            (first_facet, struct.pack("<I", 3), "invalid facet axis"),
            (first_facet_id, struct.pack("<I", nf), "surface facet index"),
            (len(original) - 2, b"", "remaining file size"),
            (len(original), b"x", "trailing data"),
        ):
            changed = original[:offset] + value
            if value and offset < len(original):
                changed += original[offset + len(value) :]
            bad = Path(tmp) / "invalid.mherom"
            bad.write_bytes(changed)
            subprocess.run([check, "--reject", str(bad), reason], check=True, env=env)

        modified = Path(tmp) / "modified.xml"
        source_free = (
            xml.read_text(encoding="utf-8")
            .replace("<VolumetricHeatSource>5", "<VolumetricHeatSource>0")
            .replace("<VolumetricHeatSource>8", "<VolumetricHeatSource>0")
        )
        modified.write_text(source_free, encoding="utf-8")
        subprocess.run(
            [*extract[:3], str(modified), str(bad), *extract[4:], "--no-point-probes"],
            env=env,
            check=True,
        )
        subprocess.run([check, "--no-channels", str(bad)], check=True, env=env)

        fixed_flux = (
            '<OtherThermalBondary i:type="SecondTypeThermalBoundary">'
            '<a:HeatFlux xmlns:a="urn:fields">1</a:HeatFlux></OtherThermalBondary>'
        )
        modified.write_text(
            source_free.replace("</Structure>", fixed_flux + "</Structure>"),
            encoding="utf-8",
        )
        rejected = subprocess.run(
            [*extract[:3], str(modified), str(bad), *extract[4:], "--no-point-probes"],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected.returncode and "nonzero Neumann" in rejected.stderr, (
            rejected.stderr
        )
        subprocess.run([check, "--no-channels", str(bad)], check=True, env=env)


if __name__ == "__main__":
    main()
