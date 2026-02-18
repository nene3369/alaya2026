"""Pure-Python scipy shim -- minimal but functional.

Provides:
  scipy.sparse  (csr_matrix, issparse, triu)
  scipy.optimize (minimize)
"""

from scipy import sparse  # noqa: F401
from scipy import optimize  # noqa: F401
