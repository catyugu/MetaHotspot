import os
import subprocess
import sys

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(script_dir, "build")

    # Build first
    print("Building project...")
    build_result = subprocess.run(
        ["cmake", "--build", build_dir, "--parallel", "--config", "Release"],
        capture_output=True, text=True
    )
    if build_result.returncode != 0:
        print("Build failed:")
        print(build_result.stdout)
        print(build_result.stderr)
        sys.exit(1)
    print("Build succeeded.")

    # Run tests
    print("Running tests...")
    test_result = subprocess.run(
        ["ctest", "--test-dir", build_dir, "--output-on-failure", "-C", "Release"],
        capture_output=True, text=True
    )
    print(test_result.stdout)
    if test_result.stderr:
        print(test_result.stderr)

    sys.exit(test_result.returncode)

if __name__ == "__main__":
    main()