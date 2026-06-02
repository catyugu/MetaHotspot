import os
import subprocess


def main():
    os.system("cmake --build build --parallel")
    subprocess.run(
        [
            "build/bin/metahotspot.exe",
            "./cases/original_steady_tests/case1.xml",
            "./results/original_steady_tests/case1_output.vtu",
            "./results/original_steady_tests/case1_output.xml",
        ]
    )
    subprocess.run(
        [
            "build/bin/metahotspot.exe",
            "./cases/original_steady_tests/case2.xml",
            "./results/original_steady_tests/case2_output.vtu",
            "./results/original_steady_tests/case2_output.xml",
        ]
    )
    subprocess.run(
        [
            "build/bin/metahotspot.exe",
            "./cases/original_steady_tests/case3.xml",
            "./results/original_steady_tests/case3_output.vtu",
            "./results/original_steady_tests/case3_output.xml",
        ]
    )
    os.system("python ./compare_steady_results.py")


if __name__ == "__main__":
    main()
