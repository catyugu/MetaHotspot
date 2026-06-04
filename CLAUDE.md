# Repository Guidelines

## Agent skills

### Issue tracker

Local markdown issues live in `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses canonical strings (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` at repo root + `docs/adr/`. See `docs/agents/domain.md`.

## Activity Tracking (Required)

- Summaries should include what changed, files touched, and any notable decisions.
- Use the scratchpad tool for follow-ups or TODOs discovered during work.

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

# !! Switch to a more trivial and clean env, so as to guarantee no dependencies on specific runtime 
conda activate cpp_env  
# Run tests
python run_tests.py

# Run cases
python run_cases.py
```

## Testing Guidelines

- **Enforce TDD for every behavior change**: follow `red -> green -> refactor`.
- **Start with verifiable baseline**: run the relevant existing tests before edits, and record the exact command + outcome in the PR/commit notes.
- **Test updates goes first**: Add or update a failing test first that reproduces the bug or captures the new requirement; implement code only after the test fails for the expected reason.
- **Keep tests green**:  Never commit if there are failing tests.
- **Regression test**: Every bug fix must include a regression test that fails before the fix and passes after it.

## Commit & Pull Request Guidelines

- Use Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`) and keep messages imperative.
- PRs: include a short summary, exact test command(s) run, and call out any changes to on-disk memory formats or `qmd` behavior.
