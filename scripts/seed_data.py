"""
种子数据脚本 - 初始化默认的 Prompt 模板和平台配置
用法: python scripts/seed_data.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models import PromptTemplate, Platform


def seed_platforms(db):
    """初始化默认平台"""
    print("正在初始化平台数据...")

    platforms_data = [
        # 社交媒体
        {"code": "xiaohongshu", "name": "小红书", "category": "social_media", "is_mock": True, "is_active": True},
        {"code": "wechat", "name": "微信公众号", "category": "social_media", "is_mock": True, "is_active": True},
        {"code": "zhihu", "name": "知乎", "category": "social_media", "is_mock": True, "is_active": True},
        # 创意写作
        {"code": "fiction", "name": "小说创作", "category": "creative_writing", "is_mock": True, "is_active": True},
        {"code": "storyboard", "name": "分镜脚本", "category": "creative_writing", "is_mock": True, "is_active": True},
        {"code": "script", "name": "剧本创作", "category": "creative_writing", "is_mock": True, "is_active": True},
        # 营销内容
        {"code": "copywriting", "name": "营销文案", "category": "marketing", "is_mock": True, "is_active": True},
        {"code": "newsletter", "name": "邮件通讯", "category": "marketing", "is_mock": True, "is_active": True},
        # 技术内容
        {"code": "technical_doc", "name": "技术文档", "category": "technical", "is_mock": True, "is_active": True},
        {"code": "tutorial", "name": "教程指南", "category": "technical", "is_mock": True, "is_active": True},
    ]

    for p_data in platforms_data:
        existing = db.query(Platform).filter(Platform.code == p_data["code"]).first()
        if not existing:
            platform = Platform(**p_data)
            db.add(platform)
            print(f"  添加平台: {p_data['name']}")

    db.commit()
    print("平台数据初始化完成！")


def seed_templates(db):
    """初始化默认 Prompt 模板"""
    print("正在初始化 Prompt 模板...")

    templates_data = [
        # 社交媒体
        {"name": "小红书爆款知识贴", "platform_code": "xiaohongshu", "description": "生成吸引眼球的小红书风格知识干货笔记，带 emoji 和行动号召", "is_public": True, "use_count": 0},
        {"name": "小红书标题党", "platform_code": "xiaohongshu", "description": "专门生成吸引点击的小红书爆款标题", "is_public": True, "use_count": 0},
        {"name": "公众号深度长文", "platform_code": "wechat", "description": "结构清晰、有深度的微信公众号文章", "is_public": True, "use_count": 0},
        {"name": "知乎专业回答", "platform_code": "zhihu", "description": "知乎风格的问答和文章格式，专业但易懂", "is_public": True, "use_count": 0},
        # 创意写作
        {"name": "短篇小说模板", "platform_code": "fiction", "description": "适合创作短篇故事的模板，有人物设定和情节框架", "is_public": True, "use_count": 0},
        {"name": "影视分镜模板", "platform_code": "storyboard", "description": "专业的镜头语言分镜脚本模板", "is_public": True, "use_count": 0},
        {"name": "剧本创作模板", "platform_code": "script", "description": "标准剧本格式，含场景、对白、动作描述", "is_public": True, "use_count": 0},
        # 营销内容
        {"name": "营销文案模板", "platform_code": "copywriting", "description": "高转化率营销文案模板，突出卖点和行动号召", "is_public": True, "use_count": 0},
        {"name": "邮件通讯模板", "platform_code": "newsletter", "description": "Newsletter 风格邮件模板，适合定期发送", "is_public": True, "use_count": 0},
        # 技术内容
        {"name": "技术文档模板", "platform_code": "technical_doc", "description": "标准技术文档格式，含概述、前置条件、示例", "is_public": True, "use_count": 0},
        {"name": "教程指南模板", "platform_code": "tutorial", "description": "循序渐进的教程格式，适合教学类内容", "is_public": True, "use_count": 0},
    ]

    for t_data in templates_data:
        existing = db.query(PromptTemplate).filter(
            PromptTemplate.name == t_data["name"],
            PromptTemplate.user_id == None
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
    finally:
        db.close()


if __name__ == "__main__":
    main()
