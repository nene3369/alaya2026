"""numpy.random sub-package shim -- re-exports from parent."""
from numpy import _Generator, _RandomState, random as _rmod

default_rng = _rmod.default_rng
RandomState = _rmod.RandomState
rand = _rmod.rand
randn = _rmod.randn
