"""Tests for lmm.sangha — Distributed Sangha P2P Protocol (僧伽)."""

from __future__ import annotations

import asyncio

from lmm.sangha.protocol import (
    MessageType,
    SanghaMessage,
    SanghaNode,
    SanghaNetwork,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tests: SanghaMessage
# ---------------------------------------------------------------------------

class TestSanghaMessage:
    def test_round_trip_json(self):
        msg = SanghaMessage(
            msg_type=MessageType.ANNOUNCE.value,
            sender_id="node-1",
            payload={"host": "127.0.0.1", "port": 8421},
        )
        json_str = msg.to_json()
        msg2 = SanghaMessage.from_json(json_str)
        assert msg2.msg_type == msg.msg_type
        assert msg2.sender_id == msg.sender_id
        assert msg2.payload == msg.payload

    def test_to_bytes(self):
        msg = SanghaMessage(
            msg_type=MessageType.HEARTBEAT.value,
            sender_id="node-2",
            payload={},
        )
        data = msg.to_bytes()
        length = int.from_bytes(data[:4], "big")
        assert length == len(data) - 4
        import json
        payload = json.loads(data[4:].decode("utf-8"))
        assert payload["sender_id"] == "node-2"


# ---------------------------------------------------------------------------
# Tests: SanghaNode (unit-level)
# ---------------------------------------------------------------------------

class TestSanghaNode:
    def test_initialization(self):
        node = SanghaNode(node_id="test-node", port=9000, specialty="wisdom")
        assert node.node_id == "test-node"
        assert node.port == 9000
        assert node.specialty == "wisdom"
        assert node.peer_count == 0

    def test_status(self):
        node = SanghaNode(node_id="test-node", port=9001)
        status = node.status()
        assert status["node_id"] == "test-node"
        assert status["running"] is False
        assert status["peer_count"] == 0

    def test_event_handler_registration(self):
        node = SanghaNode()
        handler_called = []

        async def handler(msg):
            handler_called.append(msg)

        node.on(MessageType.ANNOUNCE.value, handler)
        assert len(node._handlers[MessageType.ANNOUNCE.value]) == 1


# ---------------------------------------------------------------------------
# Tests: SanghaNetwork (unit-level, direct handler tests)
# ---------------------------------------------------------------------------

class TestSanghaNetwork:
    def test_create_two_nodes(self):
        async def _test():
            network = SanghaNetwork()
            try:
                node1 = await network.create_node(port=19421, specialty="wisdom")
                node2 = await network.create_node(port=19422, specialty="precepts")
                assert len(network.nodes) == 2
                assert node1.specialty == "wisdom"
                assert node2.specialty == "precepts"
            finally:
                await network.shutdown()
        _run(_test())

    def test_node_announces(self):
        async def _test():
            node = SanghaNode(node_id="receiver", port=19430)
            msg = SanghaMessage(
                msg_type=MessageType.ANNOUNCE.value,
                sender_id="sender-1",
                payload={"host": "127.0.0.1", "port": 19431, "specialty": "test"},
            )
            await node._handle_announce(msg)
            assert "sender-1" in node.peers
            assert node.peers["sender-1"].specialty == "test"
        _run(_test())

    def test_pattern_sharing(self):
        async def _test():
            node = SanghaNode(node_id="receiver", port=19432)
            patterns = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            msg = SanghaMessage(
                msg_type=MessageType.PATTERN_SHARE.value,
                sender_id="sender-2",
                payload={"patterns": patterns},
            )
            await node._handle_pattern_share(msg)
            assert node.shared_pattern_count == 2
        _run(_test())

    def test_council_vote_aggregation(self):
        async def _test():
            node = SanghaNode(node_id="coordinator", port=19433)
            for i in range(3):
                vote_msg = SanghaMessage(
                    msg_type=MessageType.COUNCIL_VOTE.value,
                    sender_id=f"voter-{i}",
                    payload={
                        "council_id": "council-1",
                        "vote": {
                            "node_id": f"voter-{i}",
                            "verdict": "APPROVE",
                            "insight": f"Voter {i} approves.",
                        },
                    },
                )
                await node._handle_council_vote(vote_msg)

            votes = node._council_votes.get("council-1", [])
            assert len(votes) == 3
            assert all(v["verdict"] == "APPROVE" for v in votes)
        _run(_test())
