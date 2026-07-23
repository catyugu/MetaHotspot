import subprocess
import sys


def main():
    try:
        return subprocess.run(
            ["ctest", "--test-dir", "build", "--output-on-failure"],
            check=False,
        ).returncode
    except FileNotFoundError:
        print("error: ctest is not installed or not on PATH", file=sys.stderr)
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
