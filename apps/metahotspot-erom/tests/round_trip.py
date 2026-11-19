"""Exercise the independent extractor and native reader in separate processes."""

import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    (check,) = sys.argv[1:]
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with tempfile.TemporaryDirectory() as tmp:
        native = Path(tmp) / "test.mherom"
        extract = [check, "--extract", str(native)]

        for variant, reason in (
            ("trace", "depends directly on HTC"),
            ("empty", "unknown or empty HTC group"),
        ):
            rejected = subprocess.run(
                [*extract, variant],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            assert rejected.returncode and reason in rejected.stderr, rejected.stderr
        assert not native.exists()

        subprocess.run([*extract, "standard"], env=env, check=True)
        subprocess.run([check, str(native)], check=True, env=env)

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

        subprocess.run([*extract, "partial"], env=env, check=True)
        subprocess.run([*extract, "passive"], env=env, check=True)
        subprocess.run([check, "--no-channels", str(native)], check=True, env=env)
        original = native.read_bytes()
        rejected = subprocess.run(
            [*extract, "flux"],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected.returncode and "nonzero Neumann" in rejected.stderr, (
            rejected.stderr
        )
        assert native.read_bytes() == original


if __name__ == "__main__":
    main()
