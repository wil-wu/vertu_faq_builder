import logging

from httpx import AsyncClient

from .config import enhancement_service_settings

logger = logging.getLogger(__name__)


async def get_faq_answer(httpx_client: AsyncClient, question: str) -> str:
    """获取FAQ答案"""
    resp = await httpx_client.get(
        enhancement_service_settings.faq_url, params={"q": question, "limit": 1}
    )
    data = resp.json()
    logger.info(f"FAQ answer data: {data}")
    
    categories = data.get("categories", {})
    answers = [
        subitem.get("answer", "")
        for item in categories
        for subitem in item.get("items", [])
    ]
    return "\n".join(answers)
