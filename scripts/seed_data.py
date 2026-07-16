"""
种子数据脚本 - 初始化默认平台和 Prompt 模板
用法: python scripts/seed_data.py

注意：`backend/src/api/platforms.py` 中的 _ensure_default_platforms() 会在首次 API 调用时
自动创建 DEFAULT_PLATFORMS 的平台记录。此脚本主要用于批量操作和添加 PromptTemplate 种子数据。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import SessionLocal
from src.models import PromptTemplate, Platform


def seed_platforms(db):
    """初始化默认平台（全量 16 个，覆盖前端 PlatformSelector 所有 13 个 + 扩展）"""
    print("正在初始化平台数据...")

    platforms_data = [
        # ════════════════════════════════════════════════
        # 社交平台（6 个）
        # ════════════════════════════════════════════════
        {"code": "xiaohongshu", "name": "小红书", "category": "social_media", "content_genre": "text", "is_mock": True, "is_active": True},
        {"code": "wechat", "name": "微信公众号", "category": "social_media", "content_genre": "text", "is_mock": True, "is_active": True},
        {"code": "weibo", "name": "微博", "category": "social_media", "content_genre": "text", "is_mock": True, "is_active": True},
        {"code": "douyin", "name": "抖音", "category": "social_media", "content_genre": "text", "is_mock": True, "is_active": True},
        {"code": "bilibili", "name": "B站", "category": "social_media", "content_genre": "text", "is_mock": True, "is_active": True},
        {"code": "zhihu", "name": "知乎", "category": "social_media", "content_genre": "text", "is_mock": True, "is_active": True},
        # ════════════════════════════════════════════════
        # 营销体裁（5 个）
        # ════════════════════════════════════════════════
        {"code": "copywriting", "name": "产品文案", "category": "content", "content_genre": "text", "is_mock": True, "is_active": True},
        {"code": "brand-story", "name": "品牌故事", "category": "content", "content_genre": "text", "is_mock": True, "is_active": True},
        {"code": "ad-copy", "name": "广告文案", "category": "content", "content_genre": "text", "is_mock": True, "is_active": True},
        {"code": "fiction", "name": "小说创作", "category": "content", "content_genre": "text", "is_mock": True, "is_active": True},
        {"code": "script", "name": "剧本脚本", "category": "content", "content_genre": "text", "is_mock": True, "is_active": True},
        # ════════════════════════════════════════════════
        # 专业文体（5 个）
        # ════════════════════════════════════════════════
        {"code": "tech-doc", "name": "技术文档", "category": "professional", "content_genre": "text", "is_mock": True, "is_active": True},
        {"code": "tutorial", "name": "教程指南", "category": "professional", "content_genre": "text", "is_mock": True, "is_active": True},
        {"code": "email", "name": "邮件营销", "category": "professional", "content_genre": "text", "is_mock": True, "is_active": True},
        {"code": "seo", "name": "SEO文章", "category": "professional", "content_genre": "text", "is_mock": True, "is_active": True},
        {"code": "newsletter", "name": "邮件通讯", "category": "professional", "content_genre": "text", "is_mock": True, "is_active": True},
    ]

    for p_data in platforms_data:
        existing = db.query(Platform).filter(Platform.code == p_data["code"]).first()
        if not existing:
            platform = Platform(**p_data)
            db.add(platform)
            print(f"  添加平台: {p_data['name']} ({p_data['code']})")

    db.commit()
    print("平台数据初始化完成！")


def seed_templates(db):
    """初始化默认 Prompt 模板 — 8 个写作文体各有 1-2 条带 system_prompt 的种子模板"""
    print("正在初始化 Prompt 模板...")

    templates_data = [
        # ════════════════════════════════════════════════
        # 产品文案
        # ════════════════════════════════════════════════
        {
            "name": "产品核心卖点文案",
            "platform_code": "copywriting",
            "description": "提炼产品 USP，场景化表达，突出差异化优势",
            "system_prompt": "你是一个资深文案撰稿人。请创作产品卖点文案：\n\n"
                "1. **核心定位**：一句话定义产品\n"
                "2. **三大卖点**：场景化语言描述，让用户感受到价值\n"
                "3. **用户痛点**：对应解决什么问题\n"
                "4. **差异化**：与竞品相比的优势\n"
                "5. **要求**：精准有力，不说空话，用第二人称'你'",
            "temperature": 0.7, "max_tokens": 1536, "is_public": True, "use_count": 0,
        },
        # ════════════════════════════════════════════════
        # 品牌故事
        # ════════════════════════════════════════════════
        {
            "name": "品牌起源故事",
            "platform_code": "brand-story",
            "description": "品牌初心+发展历程+愿景使命，故事真诚有温度",
            "system_prompt": "你是一个品牌叙事专家。请创作品牌起源故事：\n\n"
                "1. **初心**：创始人为什么做这个品牌，有什么洞察\n"
                "2. **历程**：从0到1的关键时刻，有细节有转折\n"
                "3. **挑战**：困难和如何克服，展现品牌韧性\n"
                "4. **愿景**：品牌使命和未来目标\n"
                "5. **要求**：故事真实动人，避免商业腔，让用户产生情感共鸣",
            "temperature": 0.8, "max_tokens": 2048, "is_public": True, "use_count": 0,
        },
        # ════════════════════════════════════════════════
        # 广告文案
        # ════════════════════════════════════════════════
        {
            "name": "社交媒体广告文案",
            "platform_code": "ad-copy",
            "description": "信息流风格，1秒抓住注意力，明确 CTA",
            "system_prompt": "你是一个效果广告文案专家。请创作广告文案：\n\n"
                "1. **标题**：3秒抓住注意力，用数字/反常识/痛点驱动\n"
                "2. **正文**：简明扼要突出利益点，一句话一个信息\n"
                "3. **紧迫感**：限时/限量/稀缺性暗示\n"
                "4. **CTA**：明确告诉用户下一步做什么\n"
                "5. **要求**：不超过100字，直接有力，每句话都为转化服务",
            "temperature": 0.8, "max_tokens": 1024, "is_public": True, "use_count": 0,
        },
        # ════════════════════════════════════════════════
        # 小说创作
        # ════════════════════════════════════════════════
        {
            "name": "短篇小说创作",
            "platform_code": "fiction",
            "description": "起承转合+人物冲突+反转结局，2000-3000字",
            "system_prompt": "你是一个小说作家。请创作一个短篇小说：\n\n"
                "1. **设定**：明确时间地点人物，建立故事世界\n"
                "2. **冲突**：核心矛盾——人与人/人与环境/人与自我的冲突\n"
                "3. **发展**：情节逐步推进，节奏有张有弛\n"
                "4. **高潮**：冲突集中爆发，对人物有关键影响\n"
                "5. **结局**：可以是闭合式或开放式，但必须有情感满足感\n"
                "6. **要求**：人物有血肉，对话自然，避免模板化语言",
            "temperature": 0.9, "max_tokens": 4096, "is_public": True, "use_count": 0,
        },
        # ════════════════════════════════════════════════
        # 剧本脚本
        # ════════════════════════════════════════════════
        {
            "name": "剧本创作模板",
            "platform_code": "script",
            "description": "标准剧本格式，场景+对白+动作描述",
            "system_prompt": "你是一个编剧。请按标准剧本格式创作：\n\n"
                "1. **场景标题**：内/外景 + 地点 + 时间\n"
                "2. **场景描述**：环境、人物、动作，只写镜头能拍到的东西\n"
                "3. **对白**：人物名+对白，对白要有潜台词\n"
                "4. **要求**：展示不要告诉，用行为和对话暗示人物心理",
            "temperature": 0.8, "max_tokens": 3072, "is_public": True, "use_count": 0,
        },
        # ════════════════════════════════════════════════
        # 技术文档
        # ════════════════════════════════════════════════
        {
            "name": "技术文档模板",
            "platform_code": "tech-doc",
            "description": "标准 API/产品文档格式，准确规范",
            "system_prompt": "你是一个技术文档工程师。请编写技术文档：\n\n"
                "1. **概述**：这是做什么的？读者需要什么前提知识？\n"
                "2. **前置条件**：版本/依赖/权限等要求\n"
                "3. **核心内容**：按逻辑顺序组织，步骤清晰\n"
                "4. **示例**：代码示例完整可运行，有注释\n"
                "5. **要求**：客观精确，术语一致，用主动语态",
            "temperature": 0.5, "max_tokens": 3072, "is_public": True, "use_count": 0,
        },
        # ════════════════════════════════════════════════
        # 教程指南
        # ════════════════════════════════════════════════
        {
            "name": "教程指南模板",
            "platform_code": "tutorial",
            "description": "循序渐进，零基础也能跟下来",
            "system_prompt": "你是一个有耐心的老师。请编写教程：\n\n"
                "1. **目标**：学完后能做出什么？给读者一个可见目标\n"
                "2. **前置**：需要什么基础？假设初学者水平\n"
                "3. **分步**：每步有明确小目标+预期结果+常见错误提示\n"
                "4. **练习**：每章有动手练习巩固\n"
                "5. **总结**：回顾学到内容，提供进阶方向\n"
                "6. **要求**：用'我们'一起做，不说'显然'——初学者最困惑的地方不要跳步",
            "temperature": 0.7, "max_tokens": 3072, "is_public": True, "use_count": 0,
        },
        # ════════════════════════════════════════════════
        # 邮件营销
        # ════════════════════════════════════════════════
        {
            "name": "邮件营销模板",
            "platform_code": "email",
            "description": "高打开率+价值感+明确 CTA，像朋友来信",
            "system_prompt": "你是一个邮件营销专家。请创作邮件文案：\n\n"
                "1. **主题行**：制造好奇心/紧迫感/价值感，30-50字\n"
                "2. **问候**：简短亲切，用用户名字\n"
                "3. **正文**：价值在前，每段一句话，紧凑有力\n"
                "4. **CTA**：明确告诉用户点哪里、为什么点\n"
                "5. **要求**：像一封真人写的信，避免群发感，移动端友好",
            "temperature": 0.7, "max_tokens": 1536, "is_public": True, "use_count": 0,
        },
    ]

    for t_data in templates_data:
        existing = db.query(PromptTemplate).filter(
            PromptTemplate.name == t_data["name"],
            PromptTemplate.user_id == None,
        ).first()

        if not existing:
            template = PromptTemplate(**t_data)
            db.add(template)
            print(f"  添加模板: {t_data['name']}")

    db.commit()
    print("Prompt 模板初始化完成！")


def main():
    print("=" * 50)
    print("种子数据初始化脚本")
    print("=" * 50)

    db = SessionLocal()
    try:
        seed_platforms(db)
        seed_templates(db)
        print("\n所有种子数据初始化完成！")
        print("提示：平台数据也会在首次访问 /api/v1/platforms 时自动创建。")
    finally:
        db.close()


if __name__ == "__main__":
    main()
