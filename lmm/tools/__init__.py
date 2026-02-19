"""Tool Use (身・Kaya) — External world actuation interface.

Provides sandboxed execution of external actions (file I/O, code execution,
API calls, web browsing) so that the Active Inference loop can *act* on the
world, not merely observe it.  This completes the action-perception cycle
required for genuine Bodhisattva conduct (菩薩行).
"""

from lmm.tools.base import Tool, ToolResult, ToolRegistry, ToolSandbox
from lmm.tools.executor import ToolExecutor

__all__ = [
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "ToolSandbox",
    "ToolExecutor",
]
