"""Artifact dependency regressions; no native toolchain is needed."""

from pathlib import Path

import pytest
from check_wheel import check_version


def test_version_comparison_pads_missing_components():
    check_version("14.51", "14.51.0", Path("vcruntime140.dll"))
    with pytest.raises(ValueError, match=r"14\.52"):
        check_version("14.52", "14.51", Path("vcruntime140.dll"))


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


@pytest.mark.parametrize(
    ("platform", "tool", "required"),
    [
        ("linux", "auditwheel", ("--plat", "manylinux_2_28_x86_64")),
        (
            "darwin",
            "delocate-wheel",
            ("--require-archs", "arm64", "--require-target-macos-version", "11.0"),
        ),
        (
            "win32",
            "delvewheel",
            ("--analyze-existing", "--ignore-existing", "--no-mangle", "mhs_c_api.dll"),
        ),
    ],
)
def test_repair_enforces_platform_policy(
    monkeypatch, tmp_path, platform, tool, required
):
    import check_python

    (tmp_path / "dist").mkdir()
    (tmp_path / "dist/raw.whl").touch()
    (tmp_path / "wheelhouse").mkdir()
    (tmp_path / "wheelhouse/repaired.whl").touch()
    commands = []
    monkeypatch.setattr(check_python, "__file__", str(tmp_path / "ci/check_python.py"))
    monkeypatch.setattr(check_python.sys, "argv", ["check_python.py", "--build-only"])
    monkeypatch.setattr(check_python.sys, "platform", platform)
    monkeypatch.setattr(check_python, "run", lambda *args: commands.append(args))
    check_python.main()
    repair = next(command for command in commands if tool in command)
    assert all(arg in repair for arg in required)
    assert "check_wheel.py" in str(commands[-1][1])


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
    check_pe(tmp_path / "core.dll", {}, sdk=True)
    with pytest.raises(ValueError, match="VCOMP140D"):
        check_pe(tmp_path / "core.dll", {})
    monkeypatch.setitem(sys.modules, "pefile", pe_for(b"VCOMP140.DLL"))
    with pytest.raises(ValueError, match="VCOMP140"):
        check_pe(tmp_path / "core.dll", {}, sdk=True)


def test_wheel_rejects_vc_runtime_older_than_build_toolset(monkeypatch, tmp_path):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    from check_wheel import check_pe

    runtime = tmp_path / "vcruntime140.dll"
    runtime.touch()
    core = SimpleNamespace(
        OPTIONAL_HEADER=SimpleNamespace(MajorLinkerVersion=14, MinorLinkerVersion=51),
        DIRECTORY_ENTRY_IMPORT=[SimpleNamespace(dll=b"vcruntime140.dll")],
    )
    old = SimpleNamespace(
        VS_FIXEDFILEINFO=[SimpleNamespace(FileVersionMS=(14 << 16) + 42)]
    )
    monkeypatch.setitem(
        sys.modules,
        "pefile",
        SimpleNamespace(
            PE=lambda path: nullcontext(old if path == str(runtime) else core)
        ),
    )
    with pytest.raises(ValueError, match=r"14\.51"):
        check_pe(tmp_path / "core.dll", {runtime.name: runtime})
    old.VS_FIXEDFILEINFO[0].FileVersionMS = (14 << 16) + 51
    check_pe(tmp_path / "core.dll", {runtime.name: runtime})
