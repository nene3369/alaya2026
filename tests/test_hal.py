"""Tests for lmm.hal — Hardware Abstraction Layer (色・Rupa)."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from lmm.hal.base import DeviceType, HardwareManager
from lmm.hal.devices import CPUDevice, CUDADevice, detect_devices


# ---------------------------------------------------------------------------
# Tests: CPUDevice
# ---------------------------------------------------------------------------

class TestCPUDevice:
    def setup_method(self):
        self.dev = CPUDevice()

    def test_capabilities(self):
        caps = self.dev.capabilities
        assert caps.device_type == DeviceType.CPU
        assert caps.device_name == "cpu"
        assert caps.max_threads >= 1

    def test_matvec_dense(self):
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        v = np.array([1.0, 0.0])
        result = self.dev.matvec(M, v)
        # Check individual elements for vendored numpy compat
        assert abs(float(result[0]) - 1.0) < 1e-6
        assert abs(float(result[1]) - 3.0) < 1e-6

    def test_matvec_sparse(self):
        M = sparse.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
        v = np.array([3.0, 4.0])
        result = self.dev.matvec(M, v)
        assert abs(float(result[0]) - 3.0) < 1e-6
        assert abs(float(result[1]) - 8.0) < 1e-6

    def test_matmul(self):
        A = np.eye(3)
        B = np.array([[1.0], [2.0], [3.0]])
        result = self.dev.matmul(A, B)
        assert abs(float(result[0, 0]) - 1.0) < 1e-6
        assert abs(float(result[1, 0]) - 2.0) < 1e-6
        assert abs(float(result[2, 0]) - 3.0) < 1e-6

    def test_tanh(self):
        x = np.array([0.0, 1.0, -1.0])
        result = self.dev.tanh(x)
        expected = np.tanh(x)
        for i in range(3):
            assert abs(float(result[i]) - float(expected[i])) < 1e-6

    def test_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        result = self.dev.softmax(x)
        total = sum(float(result[i]) for i in range(3))
        assert abs(total - 1.0) < 1e-6
        assert float(result[2]) > float(result[1]) > float(result[0])

    def test_softmax_beta(self):
        x = np.array([1.0, 2.0, 3.0])
        low_beta = self.dev.softmax(x, beta=0.1)
        high_beta = self.dev.softmax(x, beta=10.0)
        assert float(high_beta[2]) > float(low_beta[2])

    def test_solve_ode_step(self):
        V = np.array([0.0, 0.0, 0.0])
        J = np.zeros((3, 3))
        V_s = np.array([1.0, -1.0, 0.5])
        result = self.dev.solve_ode_step(V, J, V_s, G_prec=8.0, tau_leak=1.5, dt=0.01)
        assert float(result[0]) > 0
        assert float(result[1]) < 0

    def test_to_device_and_back(self):
        x = np.array([1.0, 2.0])
        on_device = self.dev.to_device(x)
        back = self.dev.to_host(on_device)
        assert abs(float(back[0]) - 1.0) < 1e-6
        assert abs(float(back[1]) - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# Tests: CUDADevice (graceful fallback when no GPU)
# ---------------------------------------------------------------------------

class TestCUDADevice:
    def test_unavailable_without_cupy(self):
        dev = CUDADevice(device_id=0)
        caps = dev.capabilities
        assert caps.device_type == DeviceType.CUDA

    def test_fallback_matvec(self):
        dev = CUDADevice(device_id=0)
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        v = np.array([1.0, 0.0])
        result = dev.matvec(M, v)
        assert abs(float(result[0]) - 1.0) < 1e-6
        assert abs(float(result[1]) - 3.0) < 1e-6


# ---------------------------------------------------------------------------
# Tests: HardwareManager
# ---------------------------------------------------------------------------

class TestHardwareManager:
    def test_detect_devices(self):
        manager = detect_devices()
        assert manager.count >= 1
        best = manager.best_device
        assert best is not None

    def test_list_devices(self):
        manager = detect_devices()
        devices = manager.list_devices()
        assert len(devices) >= 1
        cpu_found = any(d["type"] == "cpu" for d in devices)
        assert cpu_found

    def test_register_priority(self):
        manager = HardwareManager()
        cpu = CPUDevice()
        manager.register(cpu, priority=99)
        assert manager.best_device is cpu
