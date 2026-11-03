"""Artifact dependency regressions; no native toolchain is needed."""

from pathlib import Path

import pytest
from check_wheel import check_dependency, check_version


def test_runtime_in_system_library_directory_is_not_a_system_dependency(tmp_path):
    # Its build-machine location must not excuse a missing OpenMP runtime.
    with pytest.raises(ValueError, match="libgomp"):
        check_dependency("libgomp.so.1", tmp_path / "metahotspot/core.so", tmp_path, [])


def test_runtime_must_resolve_through_the_binary_search_path(tmp_path):
    package = tmp_path / "metahotspot"
    package.mkdir()
    (package / "libgomp.so.1").touch()
    check_dependency("libgomp.so.1", package / "core.so", tmp_path, [package])
    with pytest.raises(ValueError, match="libgomp"):
        check_dependency("libgomp.so.1", package / "core.so", tmp_path, [])


def test_homebrew_absolute_install_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        check_dependency(
            "/opt/homebrew/lib/libomp.dylib", tmp_path / "core", tmp_path, []
        )


def test_relative_dependency_cannot_escape_wheel(tmp_path):
    outside = tmp_path.parent / "libomp-outside.dylib"
    outside.touch()
    with pytest.raises(ValueError, match="outside"):
        check_dependency(str(outside), tmp_path / "core", tmp_path, [])


def test_binary_minimum_os_must_fit_wheel_baseline():
    check_version("11.0", "11.0", Path("libomp.dylib"))
    with pytest.raises(ValueError, match=r"26\.0"):
        check_version("26.0", "11.0", Path("libomp.dylib"))


def test_clang_cl_runtime_is_discovered_from_import_library(tmp_path):
    import subprocess

    root = Path(__file__).resolve().parents[1]
    library = tmp_path / "lib"
    binary = tmp_path / "bin"
    library.mkdir()
    binary.mkdir()
    (library / "libomp.lib").write_bytes(b"\x00libomp140.x86_64.dll\x00")
    (binary / "libomp140.x86_64.dll").touch()
    script = tmp_path / "runtime.cmake"
    script.write_text(
        f"set(WIN32 ON)\n"
        f"set(CMAKE_CXX_COMPILER_ID Clang)\n"
        f'set(CMAKE_CXX_COMPILER "{binary.as_posix()}/clang-cl.exe")\n'
        f'set(OpenMP_CXX_LIBRARIES "{library.as_posix()}/libomp.lib")\n'
        f'include("{root.as_posix()}/cmake/RuntimeDlls.cmake")\n'
        'if(NOT _mhs_openmp_runtime MATCHES "libomp140.x86_64.dll$")\n'
        '  message(FATAL_ERROR "LLVM OpenMP DLL missing")\n'
        "endif()\n"
    )
    subprocess.run(["cmake", "-P", str(script)], check=True)


def test_windows_repair_resolves_existing_wheel_dlls(monkeypatch, tmp_path):
    import check_python

    (tmp_path / "dist").mkdir()
    (tmp_path / "dist/raw.whl").touch()
    (tmp_path / "wheelhouse").mkdir()
    (tmp_path / "wheelhouse/repaired.whl").touch()
    commands = []
    monkeypatch.setattr(check_python, "__file__", str(tmp_path / "ci/check_python.py"))
    monkeypatch.setattr(check_python.sys, "argv", ["check_python.py", "--build-only"])
    monkeypatch.setattr(check_python.sys, "platform", "win32")
    monkeypatch.setattr(check_python, "run", lambda *args: commands.append(args))
    check_python.main()
    repair = next(command for command in commands if "delvewheel" in command)
    assert "--analyze-existing" in repair
    assert "--ignore-existing" in repair


def test_debug_sdk_allows_toolchain_openmp_but_release_requires_bundling(
    monkeypatch, tmp_path
):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    from check_wheel import check_pe

    def pe_for(name):
        return SimpleNamespace(
            PE=lambda _: nullcontext(
                SimpleNamespace(DIRECTORY_ENTRY_IMPORT=[SimpleNamespace(dll=name)])
            )
        )

    monkeypatch.setenv("SYSTEMROOT", str(tmp_path))
    monkeypatch.setitem(sys.modules, "pefile", pe_for(b"VCOMP140D.DLL"))
    check_pe(tmp_path / "core.dll", tmp_path, sdk=True)
    with pytest.raises(ValueError, match="VCOMP140D"):
        check_pe(tmp_path / "core.dll", tmp_path)
    monkeypatch.setitem(sys.modules, "pefile", pe_for(b"VCOMP140.DLL"))
    with pytest.raises(ValueError, match="VCOMP140"):
        check_pe(tmp_path / "core.dll", tmp_path, sdk=True)
