import os
import subprocess
import sys


def announce(msg):
    print(msg, flush=True)


def main():
    # ==============================================================================
    # 稳态 case 系列：与原版完全一致
    # ==============================================================================
    announce(">>> Running steady case: case1")
    subprocess.run(
        [
            "build/bin/metahotspot.exe",
            "./cases/simple_steady_tests/case1.xml",
            "./results/simple_steady_tests/case1_output.vtu",
            "./results/simple_steady_tests/case1_output.xml",
        ]
    )
    announce(">>> Running steady case: case2")
    subprocess.run(
        [
            "build/bin/metahotspot.exe",
            "./cases/simple_steady_tests/case2.xml",
            "./results/simple_steady_tests/case2_output.vtu",
            "./results/simple_steady_tests/case2_output.xml",
        ]
    )
    announce(">>> Running steady case: case3")
    subprocess.run(
        [
            "build/bin/metahotspot.exe",
            "./cases/simple_steady_tests/case3.xml",
            "./results/simple_steady_tests/case3_output.vtu",
            "./results/simple_steady_tests/case3_output.xml",
        ]
    )
    os.system("python ./scripts/compare_steady_results.py")

    # ==============================================================================
    # 瞬态 case 系列：观察点探针 + 末步温度场
    # ==============================================================================
    transient_dir = "./cases/simple_transient_tests"
    if os.path.isdir(transient_dir):
        # 在调用前确保 results 目录存在
        os.makedirs("./results/simple_transient_tests", exist_ok=True)

        for case_file in sorted(os.listdir(transient_dir)):
            if not case_file.endswith(".xml"):
                continue
            case_name = case_file[:-4]  # strip ".xml"
            input_path = os.path.join(transient_dir, case_file)
            vtu_path = f"./results/simple_transient_tests/{case_name}_output.vtu"
            xml_path = f"./results/simple_transient_tests/{case_name}_output.xml"
            announce(f">>> Running transient case: {case_name}")
            subprocess.run(
                [
                    "build/bin/metahotspot.exe",
                    input_path,
                    vtu_path,
                    xml_path,
                ]
            )

        os.system("python ./scripts/compare_transient_results.py")


if __name__ == "__main__":
    main()
