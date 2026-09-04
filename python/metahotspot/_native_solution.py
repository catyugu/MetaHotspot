from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from metahotspot._error import check
from metahotspot.types import MhsCompiled, MhsSolution, MhsSolutionInfo


@dataclass(frozen=True)
class ProbeSnapshot:
    name: str
    times: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class SolutionSnapshot:
    time: float
    fvm_count: int
    state: np.ndarray
    history_times: np.ndarray
    state_history: np.ndarray
    probes: list[ProbeSnapshot]


def _text(value: str | None) -> bytes | None:
    return None if value is None else value.encode("utf-8")


def _double_ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


# ---------------------------------------------------------------------------


def solve_options(dll, overrides: dict):
    """Build a C solve-options struct with *overrides* applied over defaults.

    Returns the struct object; callers must keep it alive until the matching
    solve call completes.
    """
    from metahotspot.types import _SolveOptionsCStruct

    opts = _SolveOptionsCStruct()
    dll.mhs_solve_options_default(ctypes.byref(opts))
    for name, value in overrides.items():
        setattr(opts, name, value)
    return opts


def solve(dll, compiled_handle, state, opts_overrides: dict):
    opts = solve_options(dll, opts_overrides)
    state_ptr = _double_ptr(state) if state is not None else None
    state_count = state.size if state is not None else 0
    out = ctypes.POINTER(MhsSolution)()
    check(
        dll.mhs_compiled_solve(
            compiled_handle,
            state_ptr,
            state_count,
            ctypes.byref(opts),
            ctypes.byref(out),
        ),
        "solve",
    )
    return out


def solution_snapshot(dll, handle) -> SolutionSnapshot:
    """Snapshot a native solution into Python-owned arrays."""
    info = MhsSolutionInfo()
    check(dll.mhs_solution_get_info(handle, ctypes.byref(info)), "solution_info")

    state = np.empty(int(info.state_count), dtype=np.float64)
    history_times = np.empty(int(info.record_count), dtype=np.float64)
    history_states = np.empty(
        int(info.record_count) * int(info.state_count), dtype=np.float64
    )
    check(
        dll.mhs_solution_copy_state(handle, _double_ptr(state), state.size),
        "solution_copy_state",
    )
    check(
        dll.mhs_solution_copy_history_times(
            handle, _double_ptr(history_times), history_times.size
        ),
        "solution_copy_history_times",
    )
    check(
        dll.mhs_solution_copy_history_states(
            handle, _double_ptr(history_states), history_states.size
        ),
        "solution_copy_history_states",
    )

    probes: list[ProbeSnapshot] = []
    for index in range(int(info.probe_count)):
        name_size = ctypes.c_size_t()
        record_count = ctypes.c_size_t()
        check(
            dll.mhs_solution_probe_get_info(
                handle, index, ctypes.byref(name_size), ctypes.byref(record_count)
            ),
            "solution_probe_info",
        )
        name = ctypes.create_string_buffer(name_size.value)
        times = np.empty(record_count.value, dtype=np.float64)
        values = np.empty(record_count.value, dtype=np.float64)
        check(
            dll.mhs_solution_copy_probe(
                handle,
                index,
                name,
                name_size.value,
                _double_ptr(times),
                _double_ptr(values),
                record_count.value,
            ),
            "solution_copy_probe",
        )
        probes.append(ProbeSnapshot(name.value.decode("utf-8"), times, values))

    return SolutionSnapshot(
        time=float(info.time),
        fvm_count=int(info.fvm_count),
        state=state,
        history_times=history_times,
        state_history=history_states.reshape(
            int(info.record_count), int(info.state_count)
        ),
        probes=probes,
    )


def write_vtu(dll, handle, path: str) -> None:
    check(dll.mhs_solution_write_vtu(handle, _text(path)), "write_vtu")
