"""IntentDetector 追问意图识别单元测试

针对 BUG：对话 LLM 正确表达"信息不足/需要追问"时，兜底分支不应触发 pipeline。

v3 改造：移除 TriggerExtractor 对 [GENERATE] 文本标记的处理逻辑，
本测试只覆盖纯文本意图识别（追问 vs 越权自写）。
"""

from src.conversation.intent_detector import IntentDetector


class TestLooksLikeFollowupQuestion:
    """IntentDetector.looks_like_followup_question 的判定测试"""

    # ── 期望被识别为追问（True） ──

    def test_asks_about_name_and_platform(self):
        """真实场景 1：LLM 一次问主题+平台"""
        text = (
            "您好！很高兴能帮您创作推广文案！😊\n"
            "在动笔之前，我想先了解几个关键信息：\n"
            "1. 这款智能笔记应用叫什么名字？它的核心特色是什么？\n"
            "2. 目标平台是哪里？是发在小红书、微信公众号、微博？\n"
            "3. 目标读者是谁？是学生、职场白领、创作者，还是普通大众？\n"
            "您先简单说说，我们再一步步把文案打磨出来！"
        )
        assert IntentDetector.looks_like_followup_question(text) is True

    def test_asks_what_topic_then_invites_continue(self):
        """真实场景 2：LLM 询问主题并邀请继续"""
        text = (
            "您刚才说的是：\n"
            "为一款智能笔记应用写产品推广文案\n"
            "不过目前我还在收集信息阶段，还没有真正开始创作。"
            "为了帮您写出更精准的文案，想再问您两个小问题：\n"
            "1. 这款笔记应用的名称是什么？它最突出的一个核心功能是什么？\n"
            "2. 文案打算发在哪个平台？比如小红书（偏种草）、公众号（偏深度）。\n"
            "您简单说两句，我就能开始创作啦！"
        )
        assert IntentDetector.looks_like_followup_question(text) is True

    def test_just_a_question_with_polite_invite(self):
        """真实场景 3：短追问 + 邀请"""
        text = (
            "请告诉我这款应用的核心卖点是什么？"
            "另外，目标读者是哪类人群？您先简单说说。"
        )
        assert IntentDetector.looks_like_followup_question(text) is True

    # ── 期望不被识别为追问（False） ──

    def test_empty_text(self):
        """空文本"""
        assert IntentDetector.looks_like_followup_question("") is False

    def test_too_short_text(self):
        """过短文本（< 10 字符）"""
        assert IntentDetector.looks_like_followup_question("好的") is False
        assert IntentDetector.looks_like_followup_question(None) is False

    def test_meta_dump_text(self):
        """LLM 越权自写的元描述 dump（不是追问）"""
        text = (
            "## 思考：\n"
            "好的，让我开始创作一篇推广文案。\n"
            "## 正文：\n"
            "智能笔记应用，让记录更高效……"
        )
        assert IntentDetector.looks_like_followup_question(text) is False

    def test_unauthorized_full_write(self):
        """LLM 完整写一篇（不是追问）"""
        text = (
            "好的，我直接帮您写一篇！\n\n"
            "标题：AI 时代如何让效率翻倍？\n\n"
            "引言：在当今快节奏的工作环境……\n\n"
            "第一部分：智能化趋势\n"
            "随着 AI 技术的快速发展……\n\n"
            "结论：选择 AI 工具是时代的必然。"
        )
        assert IntentDetector.looks_like_followup_question(text) is False

    def test_normal_chat_greeting(self):
        """普通聊天问候（不是追问）"""
        text = "您好！很高兴见到你。请问有什么我可以帮助您的吗？"
        assert IntentDetector.looks_like_followup_question(text) is False


class TestTriggerDetectionWithFollowup:
    """综合场景：验证「LLM 追问」+「用户消息含生成词」的判定组合"""

    def test_full_text_match_with_user_intent(self):
        """完整场景：用户说'为...写...' + LLM 追问 → 整段 full_text 应该被判为追问"""
        full_text = (
            "您好！很高兴能帮您创作推广文案！😊\n"
            "在动笔之前，我想先了解几个关键信息：\n"
            "1. 这款智能笔记应用叫什么名字？\n"
            "2. 目标平台是哪里？\n"
            "您先简单说说，我们再一步步把文案打磨出来！"
        )
        # 这个 full_text 表达追问意图
        is_unauth = IntentDetector.looks_like_unauthorized_write(full_text)
        is_followup = IntentDetector.looks_like_followup_question(full_text)
        assert is_unauth is False
        assert is_followup is True


class TestIntentDetectorRemovedMarkerMethods:
    """v3 改造：移除 [GENERATE] 文本标记相关方法"""

    def test_extract_method_removed(self):
        """extract() 已被移除（不再解析文本标记）"""
        assert not hasattr(IntentDetector, "extract")

    def test_has_marker_method_removed(self):
        """has_marker() 已被移除"""
        assert not hasattr(IntentDetector, "has_marker")

    def test_strip_method_removed(self):
        """strip() 已被移除"""
        assert not hasattr(IntentDetector, "strip")

    def test_looks_like_unauthorized_write_no_longer_accepts_has_marker(self):
        """looks_like_unauthorized_write 不再接受 has_marker 参数"""
        import inspect
        sig = inspect.signature(IntentDetector.looks_like_unauthorized_write)
        params = list(sig.parameters.keys())
        # 应该是 (cls, text)，不再有 has_marker
        assert "has_marker" not in params
        assert "text" in params


class TestTriggerExtractorAlias:
    """向后兼容：TriggerExtractor 别名 = IntentDetector"""

    def test_trigger_extractor_is_intent_detector(self):
        from src.conversation.intent_detector import TriggerExtractor, IntentDetector
        assert TriggerExtractor is IntentDetector