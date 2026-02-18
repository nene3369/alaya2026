"""Pure-Python scipy.optimize shim -- minimize with SLSQP-like projected gradient.

The LMM project uses:
  minimize(fun, x0, method="SLSQP", bounds=[(0,1)]*n)

We implement a simple projected gradient descent that respects box bounds.
"""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class OptimizeResult:
    x: object  # ndarray
    fun: float
    success: bool
    nit: int
    message: str


def minimize(fun, x0, method="SLSQP", bounds=None, jac=None,
             options=None, args=(), constraints=None):
    """Minimise *fun* starting from *x0* with box *bounds*.

    Implements projected gradient descent with numerical gradients.
    Good enough for the QUBO continuous relaxation use-case.
    """
    import numpy as np

    x = x0.copy() if isinstance(x0, np.ndarray) else np.array(list(x0))
    n = len(x._data)

    lr = 0.05
    max_iter = 200
    eps = 1e-7

    if bounds is None:
        lo = [float('-inf')] * n
        hi = [float('inf')] * n
    else:
        lo = [b[0] if b[0] is not None else float('-inf') for b in bounds]
        hi = [b[1] if b[1] is not None else float('inf') for b in bounds]

    best_x = x.copy()
    best_f = float(fun(x))

    for it in range(max_iter):
        f0 = float(fun(x))

        # Numerical gradient
        grad = [0.0] * n
        for i in range(n):
            old = x._data[i]
            x._data[i] = old + eps
            fp = float(fun(x))
            x._data[i] = old - eps
            fm = float(fun(x))
            x._data[i] = old
            grad[i] = (fp - fm) / (2 * eps)

        # Gradient step with projection
        gnorm = math.sqrt(sum(g * g for g in grad))
        if gnorm < 1e-10:
            break

        for i in range(n):
            x._data[i] -= lr * grad[i]
            # project to bounds
            if x._data[i] < lo[i]:
                x._data[i] = lo[i]
            if x._data[i] > hi[i]:
                x._data[i] = hi[i]

        f1 = float(fun(x))
        if f1 < best_f:
            best_f = f1
            best_x = x.copy()

        # Simple adaptive step
        if f1 >= f0:
            lr *= 0.5
            if lr < 1e-12:
                break
        else:
            lr *= 1.05

    return OptimizeResult(
        x=best_x,
        fun=best_f,
        success=True,
        nit=max_iter,
        message="Projected gradient descent completed",
    )
