"""Concrete compute devices — CPU, CUDA, and hardware auto-detection.

Implements the ComputeDevice interface for each supported backend.
The detect_devices() factory auto-probes the system and registers
all available hardware.
"""

from __future__ import annotations

import os
from typing import Any, List

import numpy as np
from scipy import sparse

from lmm.hal.base import (
    ComputeDevice,
    DeviceCapabilities,
    DeviceType,
    HardwareManager,
)


class CPUDevice(ComputeDevice):
    """CPU compute device using NumPy/SciPy BLAS."""

    def __init__(self) -> None:
        self._caps = DeviceCapabilities(
            device_type=DeviceType.CPU,
            device_name="cpu",
            compute_capability="blas",
            memory_bytes=_get_system_memory(),
            supports_fp16=False,
            supports_bf16=False,
            supports_int8=True,
            max_threads=os.cpu_count() or 1,
        )

    @property
    def capabilities(self) -> DeviceCapabilities:
        return self._caps

    def matvec(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        if sparse.issparse(matrix):
            # Vendored scipy may lack .dot(); fall back to @ operator
            try:
                return matrix.dot(vector)
            except AttributeError:
                return matrix @ vector
        return matrix @ vector

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    def tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def softmax(self, x: np.ndarray, beta: float = 1.0) -> np.ndarray:
        z = beta * x
        z = z - z.max()
        e = np.exp(z)
        return e / (e.sum() + 1e-12)

    def solve_ode_step(
        self,
        V: np.ndarray,
        J: np.ndarray,
        V_s: np.ndarray,
        G_prec: float,
        tau_leak: float,
        dt: float,
    ) -> np.ndarray:
        g = np.tanh(V)
        g_prime = 1.0 - g ** 2
        error = V_s - g

        if sparse.issparse(J):
            coupling = J.dot(g)
        else:
            coupling = J @ g

        dVdt = -V / tau_leak + g_prime * G_prec * error + coupling
        return V + dt * dVdt

    def to_device(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x)

    def to_host(self, x: Any) -> np.ndarray:
        return np.asarray(x)


class CUDADevice(ComputeDevice):
    """NVIDIA GPU compute device using CuPy.

    Falls back to unavailable if CuPy is not installed or no GPU is detected.
    """

    def __init__(self, device_id: int = 0) -> None:
        self._device_id = device_id
        self._cp = None
        self._available = False
        self._caps = DeviceCapabilities(
            device_type=DeviceType.CUDA,
            device_name=f"cuda:{device_id}",
        )

        try:
            import cupy as cp
            self._cp = cp
            with cp.cuda.Device(device_id):
                mem = cp.cuda.Device(device_id).mem_info
                props = cp.cuda.runtime.getDeviceProperties(device_id)
                self._caps = DeviceCapabilities(
                    device_type=DeviceType.CUDA,
                    device_name=f"cuda:{device_id}",
                    compute_capability=f"{props['major']}.{props['minor']}",
                    memory_bytes=mem[1],
                    supports_fp16=True,
                    supports_bf16=props.get("major", 0) >= 8,
                    supports_int8=True,
                    max_threads=1024,
                )
                self._available = True
        except Exception:
            pass

    @property
    def capabilities(self) -> DeviceCapabilities:
        return self._caps

    @property
    def is_available(self) -> bool:
        return self._available

    def matvec(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        if not self._available or self._cp is None:
            return matrix @ vector
        cp = self._cp
        with cp.cuda.Device(self._device_id):
            gm = cp.asarray(matrix)
            gv = cp.asarray(vector)
            return cp.asnumpy(gm @ gv)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if not self._available or self._cp is None:
            return a @ b
        cp = self._cp
        with cp.cuda.Device(self._device_id):
            ga = cp.asarray(a)
            gb = cp.asarray(b)
            return cp.asnumpy(ga @ gb)

    def tanh(self, x: np.ndarray) -> np.ndarray:
        if not self._available or self._cp is None:
            return np.tanh(x)
        cp = self._cp
        with cp.cuda.Device(self._device_id):
            gx = cp.asarray(x)
            return cp.asnumpy(cp.tanh(gx))

    def softmax(self, x: np.ndarray, beta: float = 1.0) -> np.ndarray:
        if not self._available or self._cp is None:
            z = beta * x
            z = z - z.max()
            e = np.exp(z)
            return e / (e.sum() + 1e-12)
        cp = self._cp
        with cp.cuda.Device(self._device_id):
            z = beta * cp.asarray(x)
            z = z - z.max()
            e = cp.exp(z)
            return cp.asnumpy(e / (e.sum() + 1e-12))

    def solve_ode_step(
        self,
        V: np.ndarray,
        J: np.ndarray,
        V_s: np.ndarray,
        G_prec: float,
        tau_leak: float,
        dt: float,
    ) -> np.ndarray:
        if not self._available or self._cp is None:
            # CPU fallback
            g = np.tanh(V)
            g_prime = 1.0 - g ** 2
            error = V_s - g
            coupling = J @ g if not sparse.issparse(J) else J.dot(g)
            dVdt = -V / tau_leak + g_prime * G_prec * error + coupling
            return V + dt * dVdt

        cp = self._cp
        with cp.cuda.Device(self._device_id):
            gV = cp.asarray(V)
            gJ = cp.asarray(J) if not sparse.issparse(J) else cp.sparse.csr_matrix(J)
            gVs = cp.asarray(V_s)
            g = cp.tanh(gV)
            g_prime = 1.0 - g ** 2
            error = gVs - g
            coupling = gJ @ g
            dVdt = -gV / tau_leak + g_prime * G_prec * error + coupling
            return cp.asnumpy(gV + dt * dVdt)

    def to_device(self, x: np.ndarray) -> Any:
        if self._available and self._cp is not None:
            return self._cp.asarray(x)
        return x

    def to_host(self, x: Any) -> np.ndarray:
        if self._cp is not None and hasattr(x, "get"):
            return x.get()
        return np.asarray(x)


def _get_system_memory() -> int:
    """Get total system memory in bytes."""
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
    return 0


def detect_devices() -> HardwareManager:
    """Auto-detect all available compute devices and return a HardwareManager.

    Probes for:
      1. CPU (always available)
      2. NVIDIA CUDA GPUs (via CuPy)

    Future: ROCm, TPU, FPGA, neuromorphic chips.
    """
    manager = HardwareManager()

    # CPU is always available — lowest priority
    cpu = CPUDevice()
    manager.register(cpu, priority=99)

    # Probe CUDA GPUs
    try:
        import cupy as cp
        n_gpus = cp.cuda.runtime.getDeviceCount()
        for i in range(n_gpus):
            cuda = CUDADevice(device_id=i)
            if cuda.is_available:
                manager.register(cuda, priority=10 + i)
    except Exception:
        pass

    return manager
