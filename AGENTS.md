# Repository Guidelines

## Build, Test, and Development Commands

- **Build Config**: C++20, MSVC `/W4 /WX /permissive- /utf-8 /bigobj`, Clang `-Werror -Wall -Wextra -Wpedantic`

```bash
# Build
conda activate numerical # For MKL
# Use Ninja
cmake -G "Ninja" -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
# or use MSBuild
cmake -S . -B build
cmake --build build --parallel --config Release

# Run tests
ctest --test-dir build

# Run cases
python run_cases.py
```

```bash
# Python package, tests and style gate
python -m build --wheel
python -m pip install dist/*.whl
python -m pytest python/tests
ruff format --check python ci
ruff check python ci
```

```bash
# C++ formatting gate, plus a sanitizer build tree (GCC or Clang only)
python ci/check_format.py
cmake -G "Ninja" -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DMHS_SANITIZE=ON
```

- **Versions**: the library version is defined once, as `MHS_VERSION_*` macros in `include/metahotspot/metahotspot_c_api.h`; CMake and the Python package derive from it.

## Testing Guidelines

- **Start with verifiable baseline**: run the relevant existing tests before edits, and record the exact command + outcome in the PR/commit notes.
- **Test updates goes first**: Add or update a failing test first that reproduces the bug or captures the new requirement; implement code only after the test fails for the expected reason.
- **Keep tests green**:  Never commit if there are failing tests.
- **Regression test**: Every bug fix must include a regression test that fails before the fix and passes after it.

## Commit & Pull Request Guidelines

- Use Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`) and keep messages imperative.
- PRs: include a short summary, exact test command(s) run, and call out any changes to on-disk memory formats or `qmd` behavior.
