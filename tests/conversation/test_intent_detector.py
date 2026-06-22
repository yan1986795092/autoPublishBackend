"""TriggerExtractor 追问意图识别单元测试

针对 BUG：对话 LLM 正确表达"信息不足/需要追问"时，兜底分支不应触发 pipeline。
"""

from src.conversation.intent_detector import TriggerExtractor


class TestLooksLikeFollowupQuestion:
    """TriggerExtractor.looks_like_followup_question 的判定测试"""

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
        assert TriggerExtractor.looks_like_followup_question(text) is True

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
        assert TriggerExtractor.looks_like_followup_question(text) is True

    def test_just_a_question_with_polite_invite(self):
        """真实场景 3：短追问 + 邀请"""
        text = (
            "请告诉我这款应用的核心卖点是什么？"
            "另外，目标读者是哪类人群？您先简单说说。"
        )
        assert TriggerExtractor.looks_like_followup_question(text) is True

    # ── 期望不被识别为追问（False） ──

    def test_empty_text(self):
        """空文本"""
        assert TriggerExtractor.looks_like_followup_question("") is False

    def test_too_short_text(self):
        """过短文本（< 10 字符）"""
        assert TriggerExtractor.looks_like_followup_question("好的") is False
        assert TriggerExtractor.looks_like_followup_question(None) is False

    def test_meta_dump_text(self):
        """LLM 越权自写的元描述 dump（不是追问）"""
        text = (
            "## 思考：\n\n对于一篇技术博客文章，标题需要足够吸引人。\n\n"
            "## 正文：\n\n这是一篇关于 Python 入门的文章..."
        )
        assert TriggerExtractor.looks_like_followup_question(text) is False

    def test_unauthorized_full_write(self):
        """LLM 直接写了一篇完整文章（无追问语气）"""
        text = (
            "标题：深度解析：人工智能在医疗领域的应用\n\n"
            "引言：人工智能正在以前所未有的速度改变着各行各业。\n\n"
            "第一章：技术演进\n\n从专家系统到深度学习...\n\n"
            "第二章：应用场景\n\n从影像识别到药物研发...\n\n"
            "结论：人工智能将与医疗深度融合..."
        )
        assert TriggerExtractor.looks_like_followup_question(text) is False

    def test_normal_chat_greeting(self):
        """普通闲聊问候（不含追问）"""
        text = "您好！有什么可以帮助您的吗？"
        # 单个问号 + 一个简单问句 → hits < 2，应该 False
        # 注意："您好"+"帮助"都没有触发任何追问模式
        assert TriggerExtractor.looks_like_followup_question(text) is False


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
        # 这个 full_text 既无 [GENERATE]，也无越权自写，但表达追问意图
        has_marker = TriggerExtractor.has_marker(full_text)
        is_unauth = TriggerExtractor.looks_like_unauthorized_write(full_text, has_marker=False)
        is_followup = TriggerExtractor.looks_like_followup_question(full_text)
        assert has_marker is False
        assert is_unauth is False
        assert is_followup is True
