"""Isolated ctypes helpers for P2-R-OPT2 diagnostic builds.

This module deliberately does not expose or call OMEGA_PROFILE_INTERNAL.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import omega_recurrent_production_bridge as production


class _OmegaRuntimeDiagnosticSnapshot(ctypes.Structure):
    _fields_ = [
        ("call_sequence", ctypes.c_uint64),
        ("worker_count", ctypes.c_size_t),
        ("dispatch_start_ns", ctypes.c_uint64),
        ("worker_start_ns", ctypes.c_uint64 * 4),
        ("worker_end_ns", ctypes.c_uint64 * 4),
        ("all_workers_done_ns", ctypes.c_uint64),
        ("final_gradient_reduction_start_ns", ctypes.c_uint64),
        ("final_gradient_reduction_end_ns", ctypes.c_uint64),
        ("return_ns", ctypes.c_uint64),
    ]


_SNAPSHOT_FUNCTION = "omega_runtime_diagnostic_snapshot"


def configure_library(path: str | Path) -> None:
    production.configure_library(path)
    library = production._library()
    snapshot = getattr(library, _SNAPSHOT_FUNCTION)
    snapshot.argtypes = [ctypes.c_void_p, ctypes.POINTER(_OmegaRuntimeDiagnosticSnapshot)]
    snapshot.restype = ctypes.c_int


def configure_runtime(num_threads: int) -> None:
    production.configure_runtime(num_threads)


def shutdown_runtime() -> None:
    production.shutdown_runtime()


def apply(*args: object):
    return production.apply(*args)


def loaded_library_path() -> Path:
    return production.loaded_library_path()


def snapshot() -> dict[str, object]:
    selected = production._selected_runtime()
    if selected is None:
        raise RuntimeError("diagnostic snapshot requires selected runtime")
    _, handle = selected
    value = _OmegaRuntimeDiagnosticSnapshot()
    status = int(getattr(production._library(), _SNAPSHOT_FUNCTION)(handle, ctypes.byref(value)))
    if status != 0:
        raise RuntimeError(f"{_SNAPSHOT_FUNCTION} failed with status {status}")
    workers = int(value.worker_count)
    return {
        "call_sequence": int(value.call_sequence),
        "worker_count": workers,
        "dispatch_start_ns": int(value.dispatch_start_ns),
        "worker_start_ns": [int(value.worker_start_ns[index]) for index in range(workers)],
        "worker_end_ns": [int(value.worker_end_ns[index]) for index in range(workers)],
        "all_workers_done_ns": int(value.all_workers_done_ns),
        "final_gradient_reduction_start_ns": int(value.final_gradient_reduction_start_ns),
        "final_gradient_reduction_end_ns": int(value.final_gradient_reduction_end_ns),
        "return_ns": int(value.return_ns),
    }
