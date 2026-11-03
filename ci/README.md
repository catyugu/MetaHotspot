# Build and distribution checks

The workflow builds C++ SDK ZIPs with GCC, Clang, MSVC, clang-cl and a
multi-configuration MSVC generator. Debug and sanitizer archives are development
artifacts. Windows clang-cl uses Release so its ZIP can be deployed with the
Visual C++ Redistributable. SDK consumers still need the compiler toolchain and
CMake dependencies exported by `MetaHotspotConfig.cmake`.

## Python wheels

One `py3-none` wheel is built per platform/architecture, then the **same artifact**
is installed on separate runners with Python 3.9 and 3.14. No PyPI or release
publishing is configured. There is no cibuildwheel dependency.

| Platform | Baseline | Repair |
| --- | --- | --- |
| Linux x86_64 | manylinux 2.28, glibc 2.28+ | `auditwheel repair --plat manylinux_2_28_x86_64` |
| Windows x86_64 | MSVC Release | `delvewheel repair --analyze-existing --ignore-existing --no-mangle mhs_c_api.dll` |
| macOS ARM64 | macOS 11.0+ | `delocate-wheel --require-archs arm64 --require-target-macos-version 11.0` |

Only the Linux CI build runs in the official manylinux container. Local builds
remain native: `python -m build` or `pip install .` does not require Docker.
Native Linux builds produce local `linux_x86_64` wheels; portable manylinux wheels
must be compiled in the baseline environment, not merely relabelled afterward.

macOS uses the latest ARM64 runner, with `MACOSX_DEPLOYMENT_TARGET=11.0` for all
compiled libraries. The shared setup action builds pinned upstream LLVM OpenMP
from source with that target. It avoids depending on the minimum OS version of a
Homebrew bottle. Delocate rewrites the Mach-O dependency names and rejects any
bundled dylib that requires a newer system. There is no maximum macOS version or
Intel macOS build.

CMake collects the runtime libraries needed for local source installs; missing
non-system dependencies are fatal. The Linux filter excludes named system
libraries, never all of `/usr/lib`. Windows stages and installs OpenMP alongside
TBB: CMake discovers MSVC's redistributable, while clang-cl reads the actual DLL
name from its OpenMP import library and locates that DLL near the compiler or
library, or on `PATH`. SDKs use the installed VC runtime; wheels additionally use
delvewheel's runtime bundling and loader patch.

The C API is loaded through `ctypes`, so delvewheel must analyze DLLs already in
the wheel. Its dependencies may be renamed, but `mhs_c_api.dll` keeps the name the
Python loader expects. Linux and macOS repair tools likewise inspect the C API
shared library, not just CPython extension modules. `--ignore-existing` lets
delvewheel resolve the TBB/OpenMP DLLs already bundled by CMake instead of
searching only the external build environment.

## Artifact acceptance

`ci/check_wheel.py` opens the **repaired wheel** and checks:

- the `py3-none` platform tag;
- ELF dependency resolution through in-wheel RPATH/RUNPATH, plus the glibc,
  libstdc++ and C++ ABI symbol ceilings for manylinux 2.28;
- ARM64 Mach-O dependency paths and every dylib's minimum system version;
- PE imports and delay imports, including bundled OpenMP, TBB and VC runtimes.

The same PE checker inspects Windows SDK ZIPs, allowing their documented VC
runtime dependency. Debug SDKs also require the toolchain's Debug OpenMP runtime;
that non-redistributable DLL is not put into Release SDKs or wheels. Missing
Release OpenMP is never excused by its presence in System32.

The wheel build job installs only the repaired artifact and runs all Python
tests. On macOS it first hides the original OpenMP installation. Independent
Python test jobs have no build tree or OpenMP development setup. Each also runs
`ci/wheel_smoke.py` twice, in separate processes, loading native code before and
after NumPy/SciPy and solving a 15,625-cell thermal problem with two OpenMP
threads. This catches dependency and loader failures as well as import-only
successes that never exercise the AMG solver.

Useful checks:

```sh
python -m pytest ci/test_check_wheel.py -q
python ci/check_python.py --build-only  # baseline build environment + repair tool
python ci/check_wheel.py wheelhouse/metahotspot-....whl
python ci/check_python.py --wheel wheelhouse/metahotspot-....whl
OMP_NUM_THREADS=2 python ci/wheel_smoke.py native-first
OMP_NUM_THREADS=2 python ci/wheel_smoke.py numpy-first
```
