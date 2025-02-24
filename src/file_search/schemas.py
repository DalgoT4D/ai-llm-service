from src.utils.custom_webhook import WebhookConfig
from typing import Optional
from pydantic import BaseModel


class FileQueryRequest(BaseModel):
    queries: list[str]
    assistant_prompt: str = None
    session_id: str
    webhook_config: Optional[WebhookConfig] = None
