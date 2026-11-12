# Repository Guidelines

## Setup, Build, Test, and Development Commands

```bash
# Only run these for fresh clone of the project. If uv is not available, you may use native venv or conda etc.
uv venv
uv sync
```

```bash
# Build using Ninja
cmake  -B build -DCMAKE_BUILD_TYPE=Release -G "Ninja"
cmake --build build --parallel

# or use MSBuild (Not quite suggested since it cannot generate compile_commands.json for llvm tools)
cmake -B build -G "Visual Studio 17 2022"  
cmake --build build --parallel --config Release

# Run tests
ctest --test-dir build

# Run cases
python run_cases.py
```

```bash
# Python package and tests
python -m build --wheel
python -m pip install dist/*.whl
python -m pytest python/tests
```

- **Formatting and lint**: GitHub Actions checks clang-format for `src`, `include` and `tests`, and ruff for `python` and `.github/scripts`. Local formatting and tests are optional development tools; no commit hook or local CI acceptance is required.

- **Versions**: the library version is defined once, as `MHS_VERSION_*` macros in `include/metahotspot/metahotspot_c_api.h`; CMake and the Python package derive from it.

## Testing Guidelines

- GitHub Actions owns cross-platform build, test and artifact acceptance; see [.github/CI.md](.github/CI.md).
- Add or update focused regression coverage when changing behavior. Existing CTest and Python tests remain available for local development.

## Commit & Pull Request Guidelines

- Use Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`) and keep messages imperative.
- PRs: summarize the change, report available validation results, and call out any changes to on-disk memory formats.
