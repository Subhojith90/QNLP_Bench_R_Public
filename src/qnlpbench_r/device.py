from __future__ import annotations

import platform
from dataclasses import dataclass, asdict
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    cuda_available: bool
    gpu_name: str | None
    gpu_total_memory_gb: float | None
    torch_version: str
    python_version: str
    platform: str
    cpu_count: int | None
    ram_gb: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_device(preference: str = "auto", require_cuda: bool = False) -> torch.device:
    """Select CPU or CUDA with clear errors when CUDA is required but missing."""
    preference = preference.lower()
    if preference not in {"auto", "cpu", "cuda"}:
        raise ValueError("device.preference must be one of: auto, cpu, cuda")
    cuda_ok = torch.cuda.is_available()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not cuda_ok:
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if require_cuda and not cuda_ok:
        raise RuntimeError("require_cuda=true but CUDA is unavailable. Use CPU or fix CUDA installation.")
    return torch.device("cuda" if cuda_ok else "cpu")


def get_device_info(device: torch.device | None = None) -> DeviceInfo:
    """Collect runtime device and host information."""
    cuda_available = torch.cuda.is_available()
    gpu_name = None
    gpu_mem = None
    if cuda_available:
        idx = torch.cuda.current_device() if device is None or device.type == "cuda" else 0
        props = torch.cuda.get_device_properties(idx)
        gpu_name = props.name
        gpu_mem = props.total_memory / (1024**3)
    ram = None
    cpu_count = None
    if psutil is not None:
        ram = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count(logical=True)
    return DeviceInfo(
        device=str(device) if device is not None else ("cuda" if cuda_available else "cpu"),
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_total_memory_gb=gpu_mem,
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        cpu_count=cpu_count,
        ram_gb=ram,
    )


def estimate_statevector_memory_gb(n_qubits: int, complex_bytes: int = 8, batch_size: int = 1) -> float:
    """Estimate bare dense statevector memory in GiB before autograd overhead."""
    return (2**int(n_qubits)) * int(complex_bytes) * int(batch_size) / (1024**3)


def cuda_memory_summary() -> dict[str, float | str | bool]:
    """Return CUDA memory summary if available."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    idx = torch.cuda.current_device()
    return {
        "cuda_available": True,
        "device_index": str(idx),
        "allocated_gb": torch.cuda.memory_allocated(idx) / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(idx) / (1024**3),
    }
