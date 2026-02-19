"""Distributed Sangha (僧伽) — P2P network protocol for multi-node councils.

Enables multiple Alaya instances running on different machines to form a
distributed council (真のサンガ), exchanging experience (業) over the
network and reaching consensus without a single point of failure.
This realizes Indra's Net (因陀羅網) as a decentralized system.
"""

from lmm.sangha.protocol import SanghaNode, SanghaMessage, SanghaNetwork

__all__ = [
    "SanghaNode",
    "SanghaMessage",
    "SanghaNetwork",
]
