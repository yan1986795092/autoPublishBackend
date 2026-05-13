"""
数据库初始化脚本
用法: python scripts/init_db.py
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import engine
from app.models import Base


def init_database():
    """初始化数据库，创建所有表"""
    print("正在创建数据库表...")

    Base.metadata.create_all(bind=engine)

    print("数据库表创建成功！")
    print("已创建的表:")
    for table_name in Base.metadata.tables.keys():
        print(f"  - {table_name}")


def reset_database():
    """重置数据库（删除所有表并重新创建）"""
    print("正在重置数据库...")
    Base.metadata.drop_all(bind=engine)
    print("所有表已删除")
    init_database()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="数据库管理脚本")
    parser.add_argument("action", choices=["init", "reset"], help="操作: init (初始化) 或 reset (重置)")
    args = parser.parse_args()

    if args.action == "init":
        init_database()
    elif args.action == "reset":
        reset_database()


if __name__ == "__main__":
    main()
