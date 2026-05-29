from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/v1", tags=["models"])


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


@router.get("/models", response_model=ModelListResponse)
async def list_models(request: Request):
    from ..config import get_gateway_settings
    settings = get_gateway_settings()

    models = []
    if settings.openai_api_key:
        models.append(ModelObject(id=settings.openai_model, owned_by="openai"))
    if settings.zhipu_api_key:
        models.append(ModelObject(id=settings.zhipu_model, owned_by="zhipuai"))
    if settings.qwen_api_key:
        models.append(ModelObject(id=settings.qwen_model, owned_by="qwen"))

    return ModelListResponse(data=models)
