# Build and distribution checks

The workflow builds C++ SDK ZIPs with GCC, Clang, MSVC, clang-cl and a
multi-configuration MSVC generator. Only Release ZIPs are uploaded; Debug and
sanitizer jobs only configure, build and run CTest. Release jobs also install,
package and check Windows SDK runtime dependencies. Windows clang-cl uses
Release so its ZIP can be deployed with the
Visual C++ Redistributable. SDK consumers still need the compiler toolchain and
CMake dependencies exported by `MetaHotspotConfig.cmake`.

## Python wheels

One `py3-none` wheel is built per platform/architecture, then the **same artifact**
is installed on separate runners with Python 3.10 and 3.14. No PyPI or release
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

The macOS wheel uses the latest ARM64 runner, with
`MACOSX_DEPLOYMENT_TARGET=11.0` for all compiled libraries. Its setup action builds
pinned upstream LLVM OpenMP
from source with that target. It avoids depending on the minimum OS version of a
Homebrew bottle. Delocate rewrites the Mach-O dependency names and rejects any
bundled dylib that requires a newer system. There is no maximum macOS version or
Intel macOS build. Ordinary C++ SDK jobs use Homebrew's `libomp` and the runner's
native deployment target; their ZIP does not promise the wheel's macOS baseline.

muparser is built with `ENABLE_OPENMP=OFF`; expression evaluation already uses
TBB. AMGCL retains its explicitly linked OpenMP backend.

CMake collects the runtime libraries needed for local source installs; missing
non-system dependencies are fatal. The Linux filter excludes named system
libraries, never all of `/usr/lib`. Windows stages and installs OpenMP alongside
TBB: CMake discovers MSVC's redistributable, while clang-cl reads the actual DLL
name from its OpenMP import library and locates that DLL near the compiler or
library, or on `PATH`. SDKs use the installed VC runtime. For wheels, CMake also
stages the current compiler's CRT redistributables before dependency collection, so delvewheel
cannot accidentally select older DLLs bundled with the build Python. Only
transitively needed DLLs enter the wheel; delvewheel supplies the loader patch.

The C API is loaded through `ctypes`, so delvewheel must analyze DLLs already in
the wheel. Its dependencies may be renamed, but `mhs_c_api.dll` keeps the name the
Python loader expects. Linux and macOS repair tools likewise inspect the C API
shared library, not just CPython extension modules. `--ignore-existing` lets
delvewheel resolve the TBB/OpenMP DLLs already bundled by CMake instead of
searching only the external build environment.

## Artifact acceptance

Platform repair tools enforce their own binary policies: `auditwheel repair
--plat` rejects symbols newer than manylinux 2.28 and bundles non-system ELF
dependencies; `delocate-wheel` rewrites Mach-O paths and enforces ARM64 and the
macOS 11 baseline. The project does not duplicate their ELF/Mach-O parsers.

`.github/scripts/check_wheel.py` opens the **repaired wheel** and checks the `py3-none`
platform tag and the library filename used by the ctypes loader. On Windows it
also checks PE imports and delay imports, including bundled OpenMP, TBB and VC
runtimes; bundled VC runtime versions must meet the importing binary's linker
toolset.

The same PE checker inspects Windows SDK ZIPs, allowing their documented VC
runtime dependency. It scans DLL/EXE/PYD files and builds one DLL index per
archive, skipping SDK headers and static libraries. Non-redistributable Debug
runtimes are not put into Release SDKs or wheels. Missing Release OpenMP is never
excused by its presence in System32.

The Linux wheel job runs a native solver
smoke test inside the manylinux baseline container. The macOS wheel job runs the
same smoke test after hiding the original OpenMP installation. Independent
Python test jobs have no build tree or OpenMP development setup.
`.github/scripts/check_python.py --wheel` installs the artifact, runs the Python tests, then
runs `.github/scripts/wheel_smoke.py` twice in separate processes, loading native code before
and after NumPy/SciPy and solving a 15,625-cell thermal problem with two OpenMP
threads. This catches dependency and loader failures as well as import-only
successes that never exercise the AMG solver.

## Formatting and lint

The Linux GCC job runs clang-format in check mode for `src`, `include` and
`tests`, plus ruff lint and format checks for `python` and `.github/scripts`,
using the pinned development tools from `pyproject.toml`. These checks run once
per workflow and never rewrite sources or the Git index.

All CI-specific scripts and acceptance policy live under `.github/` and are
excluded from the sdist. GitHub Actions runs artifact acceptance; contributors
are not required to reproduce it locally or install a commit hook.
