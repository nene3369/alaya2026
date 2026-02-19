"""HAL base — abstract compute device interface.

Defines the hardware abstraction that allows the FEP solver, AlayaMemory,
and other compute-intensive components to transparently use different
hardware backends (CPU BLAS, CUDA, ROCm, TPU, neuromorphic).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

import numpy as np


class DeviceType(Enum):
    """Supported hardware device types."""
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    TPU = "tpu"
    NEUROMORPHIC = "neuromorphic"
    QUANTUM = "quantum"


@dataclass
class DeviceCapabilities:
    """Hardware capabilities and constraints."""

    device_type: DeviceType
    device_name: str
    compute_capability: str = ""
    memory_bytes: int = 0
    supports_fp16: bool = False
    supports_bf16: bool = False
    supports_int8: bool = False
    max_threads: int = 1
    max_batch_size: int = 1024
    extra: Dict[str, Any] = field(default_factory=dict)


class ComputeDevice(ABC):
    """Abstract compute device — provides matrix operations on specific hardware."""

    @property
    @abstractmethod
    def capabilities(self) -> DeviceCapabilities:
        """Return device capabilities."""

    @abstractmethod
    def matvec(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Matrix-vector product: y = M @ x."""

    @abstractmethod
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix-matrix product: C = A @ B."""

    @abstractmethod
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Element-wise tanh activation."""

    @abstractmethod
    def softmax(self, x: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """Softmax: softmax(beta * x)."""

    @abstractmethod
    def solve_ode_step(
        self,
        V: np.ndarray,
        J: np.ndarray,
        V_s: np.ndarray,
        G_prec: float,
        tau_leak: float,
        dt: float,
    ) -> np.ndarray:
        """One step of the FEP KCL ODE on this device.

        dV/dt = -V/tau_leak + g'(V) * G_prec * (V_s - g(V)) + J @ g(V)
        """

    @abstractmethod
    def to_device(self, x: np.ndarray) -> Any:
        """Transfer array to device memory."""

    @abstractmethod
    def to_host(self, x: Any) -> np.ndarray:
        """Transfer array back to host (CPU) memory."""

    @property
    def is_available(self) -> bool:
        """Check if the device is ready for computation."""
        return True


class HardwareManager:
    """Manages available compute devices and selects the optimal one.

    Auto-detects hardware at init time and provides a single ``best_device``
    property for the rest of the system to use.  Falls back to CPU if no
    accelerator is available.
    """

    def __init__(self) -> None:
        self._devices: Dict[str, ComputeDevice] = {}
        self._priority: List[str] = []

    def register(self, device: ComputeDevice, priority: int = 99) -> None:
        name = device.capabilities.device_name
        self._devices[name] = device
        self._priority.append((priority, name))
        self._priority.sort()

    def get(self, name: str) -> ComputeDevice | None:
        return self._devices.get(name)

    @property
    def best_device(self) -> ComputeDevice | None:
        """Return the highest-priority available device."""
        for _, name in self._priority:
            d = self._devices.get(name)
            if d is not None and d.is_available:
                return d
        return None

    def list_devices(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "type": d.capabilities.device_type.value,
                "available": d.is_available,
                "memory_bytes": d.capabilities.memory_bytes,
            }
            for name, d in self._devices.items()
        ]

    @property
    def count(self) -> int:
        return len(self._devices)
