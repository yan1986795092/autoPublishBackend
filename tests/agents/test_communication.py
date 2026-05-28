"""MessageBus — P2P 通信、对话通道单元测试"""
import pytest
import uuid

from src.agent.communication import MessageBus, Message, MessageType, ConversationChannel


class TestMessageModel:
    """Message 数据模型测试"""

    def test_create_request(self):
        msg = Message(
            msg_id="m1",
            from_agent="WritingAgent",
            to_agent="ReviewAgent",
            msg_type=MessageType.REQUEST,
            content={"text": "请审阅"},
            conversation_id="conv1",
        )
        assert msg.msg_id == "m1"
        assert msg.from_agent == "WritingAgent"
        assert msg.to_agent == "ReviewAgent"
        assert msg.msg_type == MessageType.REQUEST
        assert msg.in_reply_to is None

    def test_create_response(self):
        msg = Message(
            msg_id="m2",
            from_agent="ReviewAgent",
            to_agent="WritingAgent",
            msg_type=MessageType.RESPONSE,
            content={"text": "审阅完毕"},
            conversation_id="conv1",
            in_reply_to="m1",
        )
        assert msg.in_reply_to == "m1"

    def test_to_dict(self):
        msg = Message(
            msg_id="m1",
            from_agent="A",
            to_agent="B",
            msg_type=MessageType.REQUEST,
            content={"text": "hello"},
            conversation_id="c1",
        )
        d = msg.to_dict()
        assert d["msg_id"] == "m1"
        assert d["msg_type"] == "request"
        assert d["content"] == {"text": "hello"}

    def test_all_message_types(self):
        for mt in MessageType:
            msg = Message(
                msg_id="x", from_agent="A", to_agent="B",
                msg_type=mt, content={}, conversation_id="c",
            )
            assert msg.msg_type == mt


class TestMessageBus:
    """MessageBus 消息总线测试"""

    @pytest.fixture
    def bus(self):
        return MessageBus()

    @pytest.mark.asyncio
    async def test_send_and_get_pending(self, bus):
        msg = Message(
            msg_id="m1", from_agent="Writer", to_agent="Reviewer",
            msg_type=MessageType.REQUEST, content={"text": "审阅"},
            conversation_id="conv1",
        )
        await bus.send(msg)

        pending = bus.get_pending("Reviewer")
        assert len(pending) == 1
        assert pending[0].msg_id == "m1"

    @pytest.mark.asyncio
    async def test_get_pending_clears_outbox(self, bus):
        msg = Message(
            msg_id="m1", from_agent="A", to_agent="B",
            msg_type=MessageType.REQUEST, content={}, conversation_id="c",
        )
        await bus.send(msg)
        bus.get_pending("B")
        assert bus.get_pending("B") == []

    @pytest.mark.asyncio
    async def test_peek_pending_does_not_clear(self, bus):
        msg = Message(
            msg_id="m1", from_agent="A", to_agent="B",
            msg_type=MessageType.REQUEST, content={}, conversation_id="c",
        )
        await bus.send(msg)

        peeked = bus.peek_pending("B")
        assert len(peeked) == 1

        still_there = bus.get_pending("B")
        assert len(still_there) == 1

    @pytest.mark.asyncio
    async def test_has_pending(self, bus):
        assert bus.has_pending("B") is False

        msg = Message(
            msg_id="m1", from_agent="A", to_agent="B",
            msg_type=MessageType.REQUEST, content={}, conversation_id="c",
        )
        await bus.send(msg)
        assert bus.has_pending("B") is True

    @pytest.mark.asyncio
    async def test_reply(self, bus):
        original = Message(
            msg_id="m1", from_agent="Writer", to_agent="Reviewer",
            msg_type=MessageType.REQUEST, content={"text": "审阅"},
            conversation_id="conv1",
        )
        await bus.send(original)

        reply_msg = await bus.reply(original, {"text": "已审阅"}, "Reviewer")
        assert reply_msg.in_reply_to == "m1"
        assert reply_msg.from_agent == "Reviewer"
        assert reply_msg.to_agent == "Writer"
        assert reply_msg.conversation_id == "conv1"

    @pytest.mark.asyncio
    async def test_send_and_wait_success(self, bus):
        msg = Message(
            msg_id="q1", from_agent="A", to_agent="B",
            msg_type=MessageType.QUERY, content={"q": "hello"}, conversation_id="c",
        )

        async def delayed_reply():
            import asyncio
            await asyncio.sleep(0.05)
            reply = Message(
                msg_id="r1", from_agent="B", to_agent="A",
                msg_type=MessageType.RESPONSE, content={"text": "world"},
                conversation_id="c", in_reply_to="q1",
            )
            await bus.reply(msg, reply.content, "B")

        import asyncio
        asyncio.create_task(delayed_reply())
        result = await bus.send_and_wait(msg, timeout=5.0)
        assert result is not None
        assert result.content["text"] == "world"

    @pytest.mark.asyncio
    async def test_send_and_wait_timeout(self, bus):
        msg = Message(
            msg_id="q2", from_agent="A", to_agent="B",
            msg_type=MessageType.QUERY, content={"q": "hello"}, conversation_id="c",
        )
        result = await bus.send_and_wait(msg, timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_broadcast(self, bus):
        msg = Message(
            msg_id="b1", from_agent="Manager", to_agent=None,
            msg_type=MessageType.BROADCAST, content={"text": "announcement"},
            conversation_id="c",
        )
        await bus.send(msg)

        # broadcast 进入所有已有 outbox 的 Agent
        # 先给各 Agent 发消息创建 outbox
        await bus.send(Message(
            msg_id="d1", from_agent="M", to_agent="Writer",
            msg_type=MessageType.REQUEST, content={}, conversation_id="c",
        ))
        await bus.send(Message(
            msg_id="d2", from_agent="M", to_agent="Reviewer",
            msg_type=MessageType.REQUEST, content={}, conversation_id="c",
        ))

        broadcast = Message(
            msg_id="b2", from_agent="Manager", to_agent=None,
            msg_type=MessageType.BROADCAST, content={"text": "all hands"},
            conversation_id="c",
        )
        await bus.broadcast(broadcast)

        writer_msgs = bus.get_pending("Writer")
        reviewer_msgs = bus.get_pending("Reviewer")
        # Writer has d1 + broadcast
        assert any(m.msg_id == "b2" for m in writer_msgs)
        assert any(m.msg_id == "b2" for m in reviewer_msgs)

    @pytest.mark.asyncio
    async def test_conversation_history(self, bus):
        await bus.send(Message(
            msg_id="m1", from_agent="A", to_agent="B",
            msg_type=MessageType.REQUEST, content={"t": "1"}, conversation_id="c1",
        ))
        await bus.send(Message(
            msg_id="m2", from_agent="B", to_agent="A",
            msg_type=MessageType.RESPONSE, content={"t": "2"}, conversation_id="c1",
        ))
        await bus.send(Message(
            msg_id="m3", from_agent="A", to_agent="C",
            msg_type=MessageType.REQUEST, content={"t": "3"}, conversation_id="c2",
        ))

        c1 = bus.get_conversation("c1")
        assert len(c1) == 2

        c2 = bus.get_conversation("c2")
        assert len(c2) == 1

    @pytest.mark.asyncio
    async def test_messages_between(self, bus):
        await bus.send(Message(
            msg_id="m1", from_agent="A", to_agent="B",
            msg_type=MessageType.REQUEST, content={}, conversation_id="c",
        ))
        await bus.send(Message(
            msg_id="m2", from_agent="B", to_agent="A",
            msg_type=MessageType.RESPONSE, content={}, conversation_id="c",
        ))
        await bus.send(Message(
            msg_id="m3", from_agent="A", to_agent="C",
            msg_type=MessageType.REQUEST, content={}, conversation_id="c",
        ))

        between = bus.get_messages_between("A", "B")
        assert len(between) == 2

    @pytest.mark.asyncio
    async def test_subscribe_and_notify(self, bus):
        received = []

        async def handler(msg):
            received.append(msg)

        bus.subscribe("Reviewer", handler)

        msg = Message(
            msg_id="m1", from_agent="A", to_agent="Reviewer",
            msg_type=MessageType.REQUEST, content={}, conversation_id="c",
        )
        await bus.notify_subscribers(msg)
        assert len(received) == 1
        assert received[0].msg_id == "m1"

    @pytest.mark.asyncio
    async def test_subscriber_error_does_not_raise(self, bus):
        async def failing_handler(msg):
            raise ValueError("oops")

        bus.subscribe("B", failing_handler)

        msg = Message(
            msg_id="m1", from_agent="A", to_agent="B",
            msg_type=MessageType.REQUEST, content={}, conversation_id="c",
        )
        # Should not raise
        await bus.notify_subscribers(msg)

    @pytest.mark.asyncio
    async def test_clear(self, bus):
        await bus.send(Message(
            msg_id="m1", from_agent="A", to_agent="B",
            msg_type=MessageType.REQUEST, content={}, conversation_id="c",
        ))
        bus.clear()
        assert bus.get_pending("B") == []
        assert bus.get_conversation("c") == []

    def test_repr(self, bus):
        r = repr(bus)
        assert "MessageBus" in r

    @pytest.mark.asyncio
    async def test_reply_sets_event_for_send_and_wait(self, bus):
        """reply 同时触发 send_and_wait 的 event"""
        msg = Message(
            msg_id="qw", from_agent="A", to_agent="B",
            msg_type=MessageType.QUERY, content={}, conversation_id="c",
        )

        import asyncio
        async def do_reply():
            await asyncio.sleep(0.05)
            await bus.reply(msg, {"text": "response"}, "B")

        asyncio.create_task(do_reply())
        result = await bus.send_and_wait(msg, timeout=5.0)
        assert result is not None
        assert result.content["text"] == "response"


class TestConversationChannel:
    """多 Agent 对话通道测试"""

    @pytest.fixture
    def bus(self):
        return MessageBus()

    @pytest.mark.asyncio
    async def test_broadcast_to_channel(self, bus):
        channel = ConversationChannel(
            channel_id="ch1",
            participants=["Writer", "Reviewer", "SEO"],
            bus=bus,
            topic="Test topic",
        )

        sent = await channel.broadcast({"text": "hello"}, "Writer")
        assert len(sent) == 2  # to Reviewer and SEO, not self

        assert bus.has_pending("Reviewer")
        assert bus.has_pending("SEO")
        assert not bus.has_pending("Writer")

    @pytest.mark.asyncio
    async def test_send_to_specific(self, bus):
        channel = ConversationChannel(
            channel_id="ch1",
            participants=["Writer", "Reviewer"],
            bus=bus,
        )

        msg = await channel.send_to("Writer", "Reviewer", {"text": "specific"})
        assert msg.to_agent == "Reviewer"

        pending = bus.get_pending("Reviewer")
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_get_history(self, bus):
        channel = ConversationChannel(
            channel_id="ch1",
            participants=["Writer", "Reviewer", "SEO"],
            bus=bus,
        )

        await channel.broadcast({"text": "hello"}, "Writer")
        history = channel.get_history()
        assert len(history) == 2  # to Reviewer and SEO, not self

    @pytest.mark.asyncio
    async def test_get_history_for(self, bus):
        channel = ConversationChannel(
            channel_id="ch1",
            participants=["Writer", "Reviewer", "SEO"],
            bus=bus,
        )

        await channel.broadcast({"text": "hi"}, "Writer")
        writer_msgs = channel.get_history_for("Writer")
        assert len(writer_msgs) == 2  # Writer sent to both

    def test_add_remove_participant(self, bus):
        channel = ConversationChannel(
            channel_id="ch1",
            participants=["Writer", "Reviewer"],
            bus=bus,
        )

        channel.add_participant("SEO")
        assert "SEO" in channel.participants

        channel.remove_participant("Writer")
        assert "Writer" not in channel.participants

    @pytest.mark.asyncio
    async def test_message_count(self, bus):
        channel = ConversationChannel(
            channel_id="ch1",
            participants=["A", "B", "C"],
            bus=bus,
        )

        assert channel.message_count == 0
        await channel.broadcast({"text": "x"}, "A")
        assert channel.message_count == 2


class TestMessageTypeEnum:
    """MessageType 枚举测试"""

    def test_values(self):
        assert MessageType.REQUEST.value == "request"
        assert MessageType.RESPONSE.value == "response"
        assert MessageType.BROADCAST.value == "broadcast"
        assert MessageType.REVIEW_REQUEST.value == "review_request"
        assert MessageType.REVIEW_RESPONSE.value == "review_response"
        assert MessageType.SUGGESTION.value == "suggestion"
        assert MessageType.QUERY.value == "query"
        assert MessageType.NOTIFY.value == "notify"
        assert MessageType.NEGOTIATE.value == "negotiate"
        assert MessageType.CHANNEL_JOIN.value == "channel_join"
        assert MessageType.CHANNEL_LEAVE.value == "channel_leave"

    def test_unique_values(self):
        values = [mt.value for mt in MessageType]
        assert len(values) == len(set(values))
