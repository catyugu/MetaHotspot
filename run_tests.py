import os
import subprocess
import sys


def main():
    os.system("cmake --build build --parallel")
    os.system("ctest --test-dir build --output-on-failure")


if __name__ == "__main__":
    main()
