"""Activate pure-Python shims before any test imports numpy/scipy."""
from lmm._vendor import inject

inject()
