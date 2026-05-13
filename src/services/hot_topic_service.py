import random
from datetime import datetime, timedelta


# 模拟各平台热门话题数据
MOCK_TOPICS = {
    "xiaohongshu": [
        {
            "title": "2026夏季必入的10件基础款单品",
            "category": "穿搭",
            "heat": 98500,
            "trend": "up",
            "tags": ["穿搭", "夏季", "基础款"],
        },
        {
            "title": "敏感肌换季护肤流程大公开",
            "category": "美妆",
            "heat": 87200,
            "trend": "up",
            "tags": ["护肤", "敏感肌", "换季"],
        },
        {
            "title": "打工人一周便当不重样食谱",
            "category": "美食",
            "heat": 76800,
            "trend": "up",
            "tags": ["美食", "便当", "打工人"],
        },
        {
            "title": "租房改造｜3平米阳台变身小花园",
            "category": "家居",
            "heat": 65400,
            "trend": "stable",
            "tags": ["租房", "改造", "阳台"],
        },
        {
            "title": "考研经验贴｜双非逆袭985",
            "category": "学习",
            "heat": 58900,
            "trend": "up",
            "tags": ["考研", "学习", "逆袭"],
        },
        {
            "title": "极简主义｜我的断舍离清单",
            "category": "生活方式",
            "heat": 52100,
            "trend": "down",
            "tags": ["极简", "断舍离", "生活方式"],
        },
        {
            "title": "周末Citywalk路线推荐",
            "category": "旅行",
            "heat": 49800,
            "trend": "up",
            "tags": ["周末", "Citywalk", "旅行"],
        },
        {
            "title": "职场沟通｜高情商回话技巧",
            "category": "职场",
            "heat": 43200,
            "trend": "stable",
            "tags": ["职场", "沟通", "情商"],
        },
    ],
    "wechat": [
        {
            "title": "深度解读2026年经济趋势与新机遇",
            "category": "财经",
            "heat": 120000,
            "trend": "up",
            "tags": ["经济", "趋势", "投资"],
        },
        {
            "title": "AI时代下，普通人的职业转型指南",
            "category": "科技",
            "heat": 98000,
            "trend": "up",
            "tags": ["AI", "职业", "转型"],
        },
        {
            "title": "如何培养孩子的自主学习能力",
            "category": "教育",
            "heat": 87500,
            "trend": "stable",
            "tags": ["教育", "亲子", "学习"],
        },
        {
            "title": "35岁以后，我悟出的10个人生道理",
            "category": "情感",
            "heat": 76200,
            "trend": "up",
            "tags": ["人生", "感悟", "成长"],
        },
        {
            "title": "一篇文章讲透ChatGPT最新应用场景",
            "category": "科技",
            "heat": 69800,
            "trend": "up",
            "tags": ["ChatGPT", "AI", "应用"],
        },
    ],
    "zhihu": [
        {
            "title": "2026年就业形势到底怎么样？",
            "category": "职场",
            "heat": 156000,
            "trend": "up",
            "tags": ["就业", "职场", "形势"],
        },
        {
            "title": "如何看待AI绘画对原画师行业的冲击",
            "category": "科技",
            "heat": 132000,
            "trend": "up",
            "tags": ["AI", "绘画", "设计"],
        },
        {
            "title": "有哪些让你相见恨晚的学习方法？",
            "category": "学习",
            "heat": 110000,
            "trend": "stable",
            "tags": ["学习", "方法", "效率"],
        },
        {
            "title": "高考志愿填报建议：专业＞学校",
            "category": "教育",
            "heat": 96000,
            "trend": "up",
            "tags": ["高考", "志愿", "教育"],
        },
        {
            "title": "长期坚持跑步给你带来了哪些改变",
            "category": "运动",
            "heat": 85000,
            "trend": "stable",
            "tags": ["跑步", "运动", "健康"],
        },
    ],
}


class HotTopicService:
    """热点话题服务"""

    def get_hot_topics(self, platform: str | None = None, limit: int = 10) -> dict:
        """获取热门话题列表"""
        result = {}

        if platform and platform in MOCK_TOPICS:
            topics = MOCK_TOPICS[platform][:limit]
            result[platform] = [
                {**t, "heat_formatted": self._format_heat(t["heat"])}
                for t in topics
            ]
        else:
            for p, topics in MOCK_TOPICS.items():
                result[p] = [
                    {**t, "heat_formatted": self._format_heat(t["heat"])}
                    for t in topics[:limit]
                ]

        return {
            "platforms": result,
            "updated_at": datetime.utcnow().isoformat(),
        }

    def _format_heat(self, heat: int) -> str:
        if heat >= 100000:
            return f"{heat / 10000:.1f}万"
        return str(heat)

    def get_trending_categories(self) -> list[dict]:
        """获取所有热门分类"""
        categories: dict[str, int] = {}
        for platform, topics in MOCK_TOPICS.items():
            for topic in topics:
                cat = topic["category"]
                categories[cat] = categories.get(cat, 0) + topic["heat"]
        sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        return [
            {"name": name, "heat": heat, "heat_formatted": self._format_heat(heat)}
            for name, heat in sorted_cats
        ]


hot_topic_service = HotTopicService()
