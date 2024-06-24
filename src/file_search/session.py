import json
from typing import Dict
from pydantic import BaseModel

from config.redis_client import RedisClient


class OpenAISessionState(BaseModel):
    id: str
    document_id: str
    thread_id: str
    assistant_id: str
    local_fpath: str


class FileSearchSession:
    _redis_client = RedisClient.get_instance()

    @classmethod
    def set(cls, key: str, value: OpenAISessionState) -> None:
        cls._redis_client.set(key, json.dumps(value.model_dump()))

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
