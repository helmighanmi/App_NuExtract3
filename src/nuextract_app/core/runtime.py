"""Hardware/device detection shared by the health endpoint and local backend.

The module never imports PyTorch at import time: the base install (API, CLI,
tests) does not include it. Detection is cached because importing ``torch`` is
expensive.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from nuextract_app.config import Device, DType


@dataclass(frozen=True)
class RuntimeInfo:
    """What the local (in-process) backend would use on this machine."""

    torch_available: bool
    torch_version: str | None
    requested_device: str
    device: str
    requested_dtype: str
    dtype: str
    cpu_count: int
    cuda_devices: int = 0
    mps_available: bool = False
    note: str | None = None

    @property
    def accelerator(self) -> str:
        return {"cuda": "nvidia", "mps": "apple", "cpu": "none"}.get(self.device, "none")

    def to_dict(self) -> dict[str, Any]:
        return {
            "torch_available": self.torch_available,
            "torch_version": self.torch_version,
            "device": self.device,
            "dtype": self.dtype,
            "accelerator": self.accelerator,
            "cpu_count": self.cpu_count,
            "cuda_devices": self.cuda_devices,
            "mps_available": self.mps_available,
            "note": self.note,
        }


def torch_installed() -> bool:
    """Cheap check that does not import torch."""
    return importlib.util.find_spec("torch") is not None


@lru_cache(maxsize=8)
def detect_runtime(device: Device = "auto", dtype: DType = "auto") -> RuntimeInfo:
    """Resolve the effective device and dtype for the Transformers backend.

    Falls back to CPU with an explanatory note when a requested accelerator is
    unavailable, so the same configuration works on a laptop and on a GPU host.
    """
    cpu_count = os.cpu_count() or 1
    if not torch_installed():
        return RuntimeInfo(
            torch_available=False,
            torch_version=None,
            requested_device=device,
            device="cpu",
            requested_dtype=dtype,
            dtype="float32",
            cpu_count=cpu_count,
            note=(
                "torch is not installed; run `uv sync --extra local` "
                "(CPU-only Linux: `uv sync --extra cpu`)"
            ),
        )

    import torch  # local import: optional dependency

    cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    mps_available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    note: str | None = None

    if device == "cuda":
        resolved = "cuda" if cuda_devices else "cpu"
    elif device == "mps":
        resolved = "mps" if mps_available else "cpu"
    elif device == "cpu":
        resolved = "cpu"
    elif cuda_devices:
        resolved = "cuda"
    elif mps_available:
        resolved = "mps"
    else:
        resolved = "cpu"

    if resolved != device and device != "auto":
        note = f"requested device={device} is unavailable; using {resolved}"

    if dtype == "auto":
        if resolved == "cuda":
            resolved_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        elif resolved == "mps":
            resolved_dtype = "float16"
        else:
            resolved_dtype = "float32"
    else:
        resolved_dtype = dtype
        if resolved == "cpu" and dtype == "float16":
            resolved_dtype = "float32"
            note = (note + "; " if note else "") + "float16 is not supported on CPU; using float32"

    return RuntimeInfo(
        torch_available=True,
        torch_version=getattr(torch, "__version__", None),
        requested_device=device,
        device=resolved,
        requested_dtype=dtype,
        dtype=resolved_dtype,
        cpu_count=cpu_count,
        cuda_devices=cuda_devices,
        mps_available=mps_available,
        note=note,
    )


def torch_dtype(torch_module: Any, name: str) -> Any:
    """Map a dtype name to a ``torch.dtype`` without importing torch here."""
    mapping = {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }
    return mapping.get(name, torch_module.float32)
