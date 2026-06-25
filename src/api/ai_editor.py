from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.api.auth import get_current_user
from src.database import get_db
from src.services.ai_editor_service import ai_editor_service
from src.services.content_service import ContentService
from src.services.content_parser_service import content_parser_service
from src.logging_shared.config import get_logger
import time

logger = get_logger("api.ai_editor")

router = APIRouter(prefix="/contents/ai-edit", tags=["AI 编辑"])


class RewriteRequest(BaseModel):
    text: str
    instruction: str = ""


class ExpandRequest(BaseModel):
    text: str


class ImproveToneRequest(BaseModel):
    text: str
    target_tone: str = "friendly"


class SummarizeRequest(BaseModel):
    text: str


class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "en"


class CheckBannedWordsRequest(BaseModel):
    text: str
    platform: str = "xiaohongshu"


@router.post("/rewrite", response_model=dict)
async def rewrite(
    request: RewriteRequest,
    current_user = Depends(get_current_user),
):
    t0 = time.time()
    user_id = str(current_user.id) if hasattr(current_user, 'id') else str(current_user)
    action = "rewrite"
    logger.info("ai_edit_request", action=action, user=user_id,
                instruction=request.instruction[:100] if request.instruction else "",
                text_length=len(request.text))
    try:
        result = await ai_editor_service.rewrite(request.text, request.instruction)
        elapsed = round(time.time() - t0, 3)
        logger.info("ai_edit_response", action=action, user=user_id,
                    status="ok", elapsed=elapsed,
                    result_length=len(result), result_preview=result)
        return {"success": True, "data": {"result": result}}
    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        logger.error("ai_edit_response", action=action, user=user_id,
                     status="error", elapsed=elapsed, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/expand", response_model=dict)
async def expand(
    request: ExpandRequest,
    current_user = Depends(get_current_user),
):
    t0 = time.time()
    user_id = str(current_user.id) if hasattr(current_user, 'id') else str(current_user)
    action = "expand"
    logger.info("ai_edit_request", action=action, user=user_id, text_length=len(request.text))
    try:
        result = await ai_editor_service.expand(request.text)
        elapsed = round(time.time() - t0, 3)
        logger.info("ai_edit_response", action=action, user=user_id, status="ok", elapsed=elapsed, result_length=len(result), result_preview=result)
        return {"success": True, "data": {"result": result}}
    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        logger.error("ai_edit_response", action=action, user=user_id, status="error", elapsed=elapsed, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/improve-tone", response_model=dict)
async def improve_tone(
    request: ImproveToneRequest,
    current_user = Depends(get_current_user),
):
    t0 = time.time()
    user_id = str(current_user.id) if hasattr(current_user, 'id') else str(current_user)
    action = "improve_tone"
    logger.info("ai_edit_request", action=action, user=user_id, tone=request.target_tone, text_length=len(request.text))
    try:
        result = await ai_editor_service.improve_tone(request.text, request.target_tone)
        elapsed = round(time.time() - t0, 3)
        logger.info("ai_edit_response", action=action, user=user_id, status="ok", elapsed=elapsed, result_length=len(result), result_preview=result)
        return {"success": True, "data": {"result": result}}
    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        logger.error("ai_edit_response", action=action, user=user_id, status="error", elapsed=elapsed, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize", response_model=dict)
async def summarize(
    request: SummarizeRequest,
    current_user = Depends(get_current_user),
):
    try:
        result = await ai_editor_service.summarize(request.text)
        return {"success": True, "data": {"result": result}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/translate", response_model=dict)
async def translate(
    request: TranslateRequest,
    current_user = Depends(get_current_user),
):
    t0 = time.time()
    user_id = str(current_user.id) if hasattr(current_user, 'id') else str(current_user)
    action = "translate"
    logger.info("ai_edit_request", action=action, user=user_id, lang=request.target_lang, text_length=len(request.text))
    try:
        result = await ai_editor_service.translate(request.text, request.target_lang)
        elapsed = round(time.time() - t0, 3)
        logger.info("ai_edit_response", action=action, user=user_id, status="ok", elapsed=elapsed, result_length=len(result), result_preview=result)
        return {"success": True, "data": {"result": result}}
    except Exception as e:
        elapsed = round(time.time() - t0, 3)
        logger.error("ai_edit_response", action=action, user=user_id, status="error", elapsed=elapsed, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-banned-words", response_model=dict)
async def check_banned_words(
    request: CheckBannedWordsRequest,
    db: Session = Depends(get_db),
):
    """实时检测文本中的违禁词（含替换建议）"""
    try:
        service = ContentService(db)
        words = service.check_banned_words(request.text, request.platform)
        return {"success": True, "data": {"words": words, "total": len(words)}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CheckBannedWordsAiRequest(BaseModel):
    text: str
    platform: str = "xiaohongshu"


@router.post("/check-banned-words-ai", response_model=dict)
async def check_banned_words_ai(
    request: CheckBannedWordsAiRequest,
):
    """AI 检测文本中的平台违禁内容（含上下文理解，可检测隐晦表达）"""
    try:
        words = await ai_editor_service.check_banned_words_ai(request.text, request.platform)
        return {"success": True, "data": words}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SuggestTagsRequest(BaseModel):
    text: str


@router.post("/suggest-tags", response_model=dict)
async def suggest_tags(
    request: SuggestTagsRequest,
    current_user = Depends(get_current_user),
):
    """根据内容智能推荐标签"""
    try:
        tags = await ai_editor_service.suggest_tags(request.text)
        return {"success": True, "data": {"tags": tags}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class GenerateHeadlinesRequest(BaseModel):
    text: str
    count: int = 5


@router.post("/generate-headlines", response_model=dict)
async def generate_headlines(
    request: GenerateHeadlinesRequest,
    current_user = Depends(get_current_user),
):
    """根据内容生成吸引人的标题"""
    try:
        headlines = await ai_editor_service.generate_headlines(request.text, request.count)
        return {"success": True, "data": {"headlines": headlines}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AnalyzeSEORequest(BaseModel):
    text: str
    platform: str = "xiaohongshu"


@router.post("/analyze-seo", response_model=dict)
async def analyze_seo(
    request: AnalyzeSEORequest,
    current_user = Depends(get_current_user),
):
    """分析内容 SEO 质量评分"""
    try:
        score = await ai_editor_service.analyze_seo(request.text, request.platform)
        return {"success": True, "data": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AdaptContentRequest(BaseModel):
    text: str
    source_platform: str
    target_platform: str


class AnalyzeViralRequest(BaseModel):
    text: str


class ExpandKeywordsRequest(BaseModel):
    keywords: str


@router.post("/adapt-content", response_model=dict)
async def adapt_content(
    request: AdaptContentRequest,
    current_user = Depends(get_current_user),
):
    """将内容适配为目标平台风格"""
    try:
        result = await ai_editor_service.adapt_content(
            request.text, request.source_platform, request.target_platform
        )
        return {"success": True, "data": {"result": result}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/expand-keywords", response_model=dict)
async def expand_keywords(
    request: ExpandKeywordsRequest,
    current_user = Depends(get_current_user),
):
    """根据种子关键词扩展长尾关键词"""
    try:
        keywords = await ai_editor_service.expand_keywords(request.keywords)
        return {"success": True, "data": {"keywords": keywords}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-viral", response_model=dict)
async def analyze_viral(
    request: AnalyzeViralRequest,
    current_user = Depends(get_current_user),
):
    """分析内容的爆款潜质"""
    try:
        score = await ai_editor_service.analyze_viral_potential(request.text)
        return {"success": True, "data": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ParseUrlRequest(BaseModel):
    url: str


@router.post("/parse-url", response_model=dict)
async def parse_url(
    request: ParseUrlRequest,
    current_user = Depends(get_current_user),
):
    """从 URL 提取标题、正文和图片"""
    try:
        result = await content_parser_service.parse_url(request.url)
        return {"success": result["success"], "data": result.get("data"), "error": result.get("error")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
