"""AutoPublish 快速演示 — 验证各模块导入和基础功能"""
import sys


def demo_imports():
    """验证所有核心模块可以正常导入"""
    modules = [
        ("src.config", "get_settings"),
        ("src.database", "engine"),
        ("src.core.security", "hash_password"),
        ("src.core.model_router", "ModelRouter"),
        ("src.agent.state", "AgentGraphState"),
        ("src.agent.schemas", "AgentTask"),
        ("src.agent.factory", "create_orchestrator"),
        ("src.utils.evaluator", "seo_score"),
        ("src.utils.monitor", "get_monitor"),
        ("src.memory.chat_history", "ChatHistory"),
    ]

    passed = 0
    for module_path, attr in modules:
        try:
            mod = __import__(module_path, fromlist=[attr])
            getattr(mod, attr)
            print(f"  [OK] {module_path}.{attr}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {module_path}.{attr}: {e}")

    print(f"\n  Result: {passed}/{len(modules)} imports successful")
    return passed == len(modules)


def demo_chat_history():
    """演示 ChatHistory 功能"""
    from src.memory.chat_history import ChatHistory

    ch = ChatHistory(max_messages=5)
    ch.add_user("Hello, can you write an article about AI?")
    ch.add_assistant("Sure! Here's a draft about artificial intelligence...")
    ch.add_user("Make it more technical.")
    ch.add_assistant("I've revised it with more technical depth...")

    messages = ch.get_messages()
    print(f"  Messages stored: {len(messages)}")
    print(f"  Context prompt (first 100 chars): {ch.to_prompt_context(200)[:100]}...")
    return len(messages) == 4


def demo_seo_evaluator():
    """演示 SEO 评分功能"""
    from src.utils.evaluator import seo_score, check_banned_words, extract_output

    content = "这是一篇关于人工智能发展趋势的分析文章。AI技术正在改变我们的工作和生活方式..."
    keywords = [{"keyword": "人工智能", "density_target": 0.02}]
    score = seo_score(content, keywords, "AI 发展趋势分析")
    print(f"  SEO score: {score}")

    banned = check_banned_words(content)
    print(f"  Banned words found: {len(banned)}")

    output = extract_output("REASONING: This is a test.\nOUTPUT: Final article text here.")
    print(f"  Extracted output: {output[:50]}...")
    return True


def demo_monitor():
    """演示 Monitor 指标记录"""
    from src.utils.monitor import get_monitor

    monitor = get_monitor()
    monitor.record("WritingAgent", tokens=1500, elapsed_ms=3200, cache_hit=False)
    monitor.record("SEOAgent", tokens=800, elapsed_ms=1200, cache_hit=True)
    monitor.record("ReviewAgent", tokens=600, elapsed_ms=900, cache_hit=False)

    summary = monitor.summary()
    print(f"  Total requests: {summary['total_requests']}")
    print(f"  Total tokens: {summary['total_tokens']}")
    print(f"  Avg time: {summary['avg_time_ms']}ms")
    print(f"  Cache hit rate: {summary['cache_hit_rate']}")
    return summary["total_requests"] == 3


def main():
    print("=" * 50)
    print("  AutoPublish Demo — Module Verification")
    print("=" * 50)

    demos = [
        ("Import Check", demo_imports),
        ("Chat History", demo_chat_history),
        ("SEO Evaluator", demo_seo_evaluator),
        ("Monitor", demo_monitor),
    ]

    for name, fn in demos:
        print(f"\n--- {name} ---")
        try:
            fn()
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print("  Demo complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
