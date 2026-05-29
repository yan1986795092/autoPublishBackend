import uvicorn
from .server import create_gateway_app

if __name__ == "__main__":
    app = create_gateway_app()
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
