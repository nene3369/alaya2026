"""Hardware Abstraction Layer (色・Rupa) — Physical device interface.

Abstracts away the compute substrate so the FEP ODE solver can run on
CPU, GPU (CUDA/ROCm), TPU, or future neuromorphic/quantum hardware
through a unified HAL interface.
"""

from lmm.hal.base import ComputeDevice, DeviceCapabilities, HardwareManager
from lmm.hal.devices import CPUDevice, CUDADevice, detect_devices

__all__ = [
    "ComputeDevice",
    "DeviceCapabilities",
    "HardwareManager",
    "CPUDevice",
    "CUDADevice",
    "detect_devices",
]
