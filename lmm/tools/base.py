"""Tool Use base classes — sandboxed external action primitives.

Defines the Tool protocol, result container, registry, and sandbox policy
that govern how the Active Inference engine interacts with the external world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Tool result
# ---------------------------------------------------------------------------

class ToolStatus(Enum):
    """Outcome of a tool invocation."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    DENIED = "denied"


@dataclass
class ToolResult:
    """Result returned by a Tool execution.

    Carries both the raw payload and metadata that the Active Inference
    engine can convert into an ActionResult for belief updates.
    """

    status: ToolStatus
    output: Any = None
    error: str = ""
    elapsed_ms: float = 0.0
    node_indices: list[int] = field(default_factory=list)
    relevance_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Tool(Protocol):
    """Protocol that every tool must satisfy."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def category(self) -> str: ...

    async def execute(self, params: Dict[str, Any]) -> ToolResult: ...


# ---------------------------------------------------------------------------
# Sandbox policy
# ---------------------------------------------------------------------------

class SandboxLevel(Enum):
    """Sandboxing strictness."""
    UNRESTRICTED = "unrestricted"
    STANDARD = "standard"
    STRICT = "strict"


@dataclass
class SandboxPolicy:
    """Defines what a tool is allowed to do."""

    level: SandboxLevel = SandboxLevel.STANDARD
    max_execution_seconds: float = 30.0
    allowed_paths: list[str] = field(default_factory=list)
    denied_paths: list[str] = field(default_factory=lambda: [
        "/etc", "/boot", "/proc/sys", "/sys",
    ])
    allowed_hosts: list[str] = field(default_factory=list)
    denied_commands: list[str] = field(default_factory=lambda: [
        "rm -rf /", "mkfs", "dd if=", "shutdown", "reboot",
        "format c:", "> /dev/sda",
    ])
    max_output_bytes: int = 1_048_576  # 1 MiB


class ToolSandbox:
    """Enforces SandboxPolicy on tool invocations.

    Validates parameters *before* execution and truncates output *after*.
    Integrates with Upali's precept checking for ethical gating.
    """

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy()

    def validate(self, tool_name: str, params: Dict[str, Any]) -> str | None:
        """Return an error message if the invocation violates policy, else None."""
        # Check denied commands
        cmd = str(params.get("command", "")).lower()
        for denied in self.policy.denied_commands:
            if denied.lower() in cmd:
                return f"Command blocked by sandbox: '{denied}'"

        # Check denied paths
        for key in ("path", "file_path", "directory"):
            path_val = str(params.get(key, ""))
            for denied in self.policy.denied_paths:
                if path_val.startswith(denied):
                    return f"Path blocked by sandbox: '{denied}'"

        # Check allowed hosts for network tools
        host = params.get("host", params.get("url", ""))
        if host and self.policy.allowed_hosts:
            from urllib.parse import urlparse
            parsed = urlparse(str(host))
            hostname = parsed.hostname or str(host)
            if hostname not in self.policy.allowed_hosts:
                return f"Host not in allowlist: '{hostname}'"

        return None

    def truncate_output(self, output: Any) -> Any:
        """Truncate output to policy limits."""
        if isinstance(output, (str, bytes)):
            max_b = self.policy.max_output_bytes
            if len(output) > max_b:
                if isinstance(output, str):
                    return output[:max_b] + f"\n... [truncated at {max_b} bytes]"
                return output[:max_b]
        return output


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Central registry for discoverable tools.

    Tools register themselves by name; the Active Inference engine queries
    the registry to find the right tool for a given action plan.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool
        cat = tool.category
        if cat not in self._categories:
            self._categories[cat] = []
        if tool.name not in self._categories[cat]:
            self._categories[cat].append(tool.name)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        """Return a manifest of registered tools (for LLM function-calling)."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category,
            }
            for t in self._tools.values()
        ]

    def by_category(self, category: str) -> List[Tool]:
        names = self._categories.get(category, [])
        return [self._tools[n] for n in names if n in self._tools]

    @property
    def count(self) -> int:
        return len(self._tools)
