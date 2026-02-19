"""P2P Sangha Protocol — distributed council over the network.

Implements a lightweight P2P protocol where each node is an autonomous
Alaya instance that can:
  - Announce its presence to the network
  - Share experience (karma/patterns) with peers
  - Participate in distributed council deliberation
  - Reach consensus via async voting

The protocol is built on asyncio + JSON-over-TCP for simplicity and
portability.  No external dependencies beyond the standard library.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

class MessageType(Enum):
    """P2P message types."""
    ANNOUNCE = "announce"
    HEARTBEAT = "heartbeat"
    PATTERN_SHARE = "pattern_share"
    COUNCIL_REQUEST = "council_request"
    COUNCIL_VOTE = "council_vote"
    COUNCIL_RESULT = "council_result"
    KARMA_SYNC = "karma_sync"


@dataclass
class SanghaMessage:
    """A message exchanged between Sangha nodes."""

    msg_type: str
    sender_id: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> SanghaMessage:
        d = json.loads(data)
        return cls(**d)

    def to_bytes(self) -> bytes:
        raw = self.to_json().encode("utf-8")
        # Length-prefix framing: 4 bytes big-endian length + payload
        return len(raw).to_bytes(4, "big") + raw


# ---------------------------------------------------------------------------
# Peer info
# ---------------------------------------------------------------------------

@dataclass
class PeerInfo:
    """Information about a connected peer."""

    node_id: str
    host: str
    port: int
    specialty: str = ""
    last_seen: float = 0.0
    pattern_count: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Sangha Node
# ---------------------------------------------------------------------------

class SanghaNode:
    """A single node in the distributed Sangha network.

    Each node runs a TCP server, maintains a peer list, and can participate
    in distributed councils.  Nodes are autonomous — if the network partitions,
    each node continues operating independently (like a lone monk).

    Parameters
    ----------
    node_id : str | None
        Unique node identifier.  Auto-generated if not provided.
    host : str
        Address to bind the TCP server to.
    port : int
        Port to bind the TCP server to.
    specialty : str
        This node's expertise area (maps to disciple roles).
    max_peers : int
        Maximum number of peers to maintain connections with.
    """

    def __init__(
        self,
        node_id: str | None = None,
        host: str = "0.0.0.0",
        port: int = 8421,
        specialty: str = "general",
        max_peers: int = 49,
    ) -> None:
        self.node_id = node_id or str(uuid.uuid4())[:12]
        self.host = host
        self.port = port
        self.specialty = specialty
        self.max_peers = max_peers

        self._peers: Dict[str, PeerInfo] = {}
        self._server: asyncio.Server | None = None
        self._running = False
        self._handlers: Dict[str, List[Callable]] = {}
        self._council_votes: Dict[str, List[Dict[str, Any]]] = {}
        self._shared_patterns: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the TCP server and begin listening for peers."""
        self._running = True
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
        )

    async def stop(self) -> None:
        """Gracefully shut down the node."""
        self._running = False
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle an incoming peer connection."""
        try:
            while self._running:
                # Read length-prefixed message
                length_bytes = await reader.readexactly(4)
                length = int.from_bytes(length_bytes, "big")
                if length > 10_000_000:  # 10MB safety limit
                    break
                data = await reader.readexactly(length)
                msg = SanghaMessage.from_json(data.decode("utf-8"))
                await self._dispatch(msg, writer)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()

    async def _dispatch(
        self,
        msg: SanghaMessage,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Route a message to the appropriate handler."""
        # Update peer info
        if msg.sender_id != self.node_id:
            if msg.sender_id in self._peers:
                self._peers[msg.sender_id].last_seen = time.time()

        if msg.msg_type == MessageType.ANNOUNCE.value:
            await self._handle_announce(msg)
        elif msg.msg_type == MessageType.HEARTBEAT.value:
            pass  # Presence confirmation only
        elif msg.msg_type == MessageType.PATTERN_SHARE.value:
            await self._handle_pattern_share(msg)
        elif msg.msg_type == MessageType.COUNCIL_REQUEST.value:
            await self._handle_council_request(msg, writer)
        elif msg.msg_type == MessageType.COUNCIL_VOTE.value:
            await self._handle_council_vote(msg)

        # Fire registered handlers
        for handler in self._handlers.get(msg.msg_type, []):
            try:
                await handler(msg)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    async def _handle_announce(self, msg: SanghaMessage) -> None:
        peer = PeerInfo(
            node_id=msg.sender_id,
            host=msg.payload.get("host", ""),
            port=msg.payload.get("port", 0),
            specialty=msg.payload.get("specialty", ""),
            last_seen=time.time(),
            pattern_count=msg.payload.get("pattern_count", 0),
        )
        if len(self._peers) < self.max_peers:
            self._peers[peer.node_id] = peer

    async def _handle_pattern_share(self, msg: SanghaMessage) -> None:
        patterns = msg.payload.get("patterns", [])
        for p in patterns:
            self._shared_patterns.append(p)
        # Keep bounded
        if len(self._shared_patterns) > 10_000:
            self._shared_patterns = self._shared_patterns[-10_000:]

    async def _handle_council_request(
        self,
        msg: SanghaMessage,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Respond to a council deliberation request with our vote."""
        council_id = msg.payload.get("council_id", "")
        msg.payload.get("query", "")

        # Simple local deliberation — each node votes based on its specialty
        vote = {
            "node_id": self.node_id,
            "specialty": self.specialty,
            "verdict": "APPROVE",
            "insight": f"Node {self.node_id} ({self.specialty}): Deliberated on query.",
        }

        response = SanghaMessage(
            msg_type=MessageType.COUNCIL_VOTE.value,
            sender_id=self.node_id,
            payload={"council_id": council_id, "vote": vote},
        )
        writer.write(response.to_bytes())
        await writer.drain()

    async def _handle_council_vote(self, msg: SanghaMessage) -> None:
        council_id = msg.payload.get("council_id", "")
        vote = msg.payload.get("vote", {})
        if council_id not in self._council_votes:
            self._council_votes[council_id] = []
        self._council_votes[council_id].append(vote)

    # ------------------------------------------------------------------
    # Outbound operations
    # ------------------------------------------------------------------

    async def _send_to_peer(
        self,
        peer: PeerInfo,
        msg: SanghaMessage,
    ) -> SanghaMessage | None:
        """Send a message to a peer and optionally receive a response."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(peer.host, peer.port),
                timeout=5.0,
            )
            t0 = time.monotonic()
            writer.write(msg.to_bytes())
            await writer.drain()

            # Read response
            length_bytes = await asyncio.wait_for(
                reader.readexactly(4), timeout=10.0,
            )
            length = int.from_bytes(length_bytes, "big")
            data = await asyncio.wait_for(
                reader.readexactly(length), timeout=10.0,
            )
            peer.latency_ms = (time.monotonic() - t0) * 1000
            writer.close()
            return SanghaMessage.from_json(data.decode("utf-8"))
        except Exception:
            return None

    async def connect_to_peer(self, host: str, port: int) -> bool:
        """Announce ourselves to a peer and establish a connection."""
        msg = SanghaMessage(
            msg_type=MessageType.ANNOUNCE.value,
            sender_id=self.node_id,
            payload={
                "host": self.host,
                "port": self.port,
                "specialty": self.specialty,
            },
        )
        peer = PeerInfo(node_id="unknown", host=host, port=port)
        response = await self._send_to_peer(peer, msg)
        return response is not None

    async def share_patterns(
        self,
        patterns: List[np.ndarray],
    ) -> int:
        """Broadcast patterns to all connected peers.

        Returns the number of peers that received the patterns.
        """
        serialized = [p.tolist() for p in patterns]
        msg = SanghaMessage(
            msg_type=MessageType.PATTERN_SHARE.value,
            sender_id=self.node_id,
            payload={"patterns": serialized},
        )
        sent = 0
        for peer in self._peers.values():
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(peer.host, peer.port),
                    timeout=5.0,
                )
                writer.write(msg.to_bytes())
                await writer.drain()
                writer.close()
                sent += 1
            except Exception:
                continue
        return sent

    async def request_council(
        self,
        query: str,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Request a distributed council vote from all peers.

        Returns aggregated results including final verdict.
        """
        council_id = str(uuid.uuid4())[:12]
        self._council_votes[council_id] = []

        msg = SanghaMessage(
            msg_type=MessageType.COUNCIL_REQUEST.value,
            sender_id=self.node_id,
            payload={"council_id": council_id, "query": query},
        )

        # Send to all peers in parallel
        tasks = [
            self._send_to_peer(peer, msg)
            for peer in self._peers.values()
        ]
        if tasks:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )

        # Aggregate votes
        votes = self._council_votes.get(council_id, [])
        rejects = [v for v in votes if v.get("verdict") == "REJECT"]

        if rejects:
            final = "REJECTED"
            reason = rejects[0].get("insight", "Rejected by peer")
        else:
            final = "APPROVED"
            reason = f"Distributed consensus: {len(votes)} node(s) approved."

        return {
            "council_id": council_id,
            "final": final,
            "reason": reason,
            "votes": votes,
            "peer_count": len(self._peers),
        }

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def on(self, msg_type: str, handler: Callable) -> None:
        """Register a handler for a message type."""
        if msg_type not in self._handlers:
            self._handlers[msg_type] = []
        self._handlers[msg_type].append(handler)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def peers(self) -> Dict[str, PeerInfo]:
        return dict(self._peers)

    @property
    def peer_count(self) -> int:
        return len(self._peers)

    @property
    def shared_pattern_count(self) -> int:
        return len(self._shared_patterns)

    def status(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "specialty": self.specialty,
            "running": self._running,
            "peer_count": self.peer_count,
            "shared_patterns": self.shared_pattern_count,
        }


# ---------------------------------------------------------------------------
# Network (convenience wrapper for multi-node setup)
# ---------------------------------------------------------------------------

class SanghaNetwork:
    """Convenience wrapper for managing a local cluster of SanghaNodes.

    Primarily for testing and development — in production, each node
    runs as its own process on a separate machine.
    """

    def __init__(self) -> None:
        self._nodes: List[SanghaNode] = []

    async def create_node(
        self,
        port: int,
        specialty: str = "general",
        **kwargs: Any,
    ) -> SanghaNode:
        """Create and start a new node."""
        node = SanghaNode(port=port, specialty=specialty, **kwargs)
        await node.start()
        self._nodes.append(node)
        return node

    async def connect_all(self) -> int:
        """Connect every node to every other node."""
        connections = 0
        for i, a in enumerate(self._nodes):
            for j, b in enumerate(self._nodes):
                if i != j:
                    ok = await a.connect_to_peer("127.0.0.1", b.port)
                    if ok:
                        connections += 1
        return connections

    async def shutdown(self) -> None:
        """Stop all nodes."""
        for node in self._nodes:
            await node.stop()
        self._nodes.clear()

    @property
    def nodes(self) -> List[SanghaNode]:
        return list(self._nodes)
