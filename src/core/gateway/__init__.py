from .config import GatewaySettings, get_gateway_settings

# Lazy import — create_gateway_app pulls in FastAPI + observability deps
# that may not be available in all environments
__all__ = ["GatewaySettings", "get_gateway_settings", "create_gateway_app"]


def __getattr__(name: str):
    if name == "create_gateway_app":
        from .server import create_gateway_app as _fn
        return _fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
