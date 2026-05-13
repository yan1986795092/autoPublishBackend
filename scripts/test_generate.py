import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models import User
from app.core.security import create_access_token
import traceback

async def run():
    db = SessionLocal()
    try:
        user = db.query(User).first()
        if not user:
            print("No user found!")
            return
        print(f"User: {user.email}, ID: {user.id}")

        token = create_access_token(data={"sub": user.id})
        print(f"Token: {token[:20]}...")

        # Now simulate the FastAPI request
        from fastapi.testclient import TestClient
        from app.main import app

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/contents/generate",
                json={
                    "topic": "测试：职场沟通技巧",
                    "platform_code": "xiaohongshu",
                    "parameters": {
                        "length": "medium",
                        "creativity": 0.7,
                        "tone": "friendly"
                    }
                },
                headers={"Authorization": f"Bearer {token}"}
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(run())
