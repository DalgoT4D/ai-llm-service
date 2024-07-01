import json
from typing import Dict, Optional
from pydantic import BaseModel

from config.redis_client import RedisClient


class OpenAISessionState(BaseModel):
    id: str
    local_fpaths: list[str]
    document_id: Optional[str] = None
    thread_id: Optional[str] = None
    assistant_id: Optional[str] = None


class FileSearchSession:
    _redis_client = RedisClient.get_instance()

    @classmethod
    def set(cls, key: str, value: OpenAISessionState) -> OpenAISessionState:
        cls._redis_client.set(key, json.dumps(value.model_dump()))
        return value

    @classmethod
    def get(cls, key) -> OpenAISessionState:
        result = cls._redis_client.get(key)
        if result:
            return OpenAISessionState(**json.loads(result))
        return None

    @classmethod
    def get_dict(cls, key) -> Dict:
        result = cls._redis_client.get(key)
        if result:
            return json.loads(result)
        return None

    @classmethod
    def remove(cls, key) -> None:
        cls._redis_client.delete(key)
