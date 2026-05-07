import json
import time
from typing import Any, Dict, Optional
from enum import Enum
from pydantic import BaseModel

from config.redis_client import RedisClient


# How often `wait_for_session_field` peeks at the session record in redis.
POLL_INTERVAL_SECS = 3


class SessionStatusEnum(str, Enum):
    active = "active"
    locked = "locked"  # set after first query so no more files can be uploaded in the same session. 


class OpenAISessionState(BaseModel):
    id: str
    local_fpaths: list[str]
    document_ids: Optional[list[str]] = []
    status: SessionStatusEnum = SessionStatusEnum.active
    vector_store_id: Optional[str] = None    # was: assistant_id
    conversation_id: Optional[str] = None    # was: thread_id
    collection_id: Optional[str] = None      # for DELETE /collections/{id} on cleanup

    # In-flight per-query-call state. Reset by the Celery task at task start.
    queries: list[str] = []
    results: list[Optional[str]] = []        # parallel to queries; None until filled
    error_message: Optional[str] = None      # set by either webhook on failure


class FileSearchSession:
    _redis_client = RedisClient.get_instance()

    @classmethod
    def set(cls, key: str, value: OpenAISessionState) -> OpenAISessionState:
        cls._redis_client.set(key, json.dumps(value.model_dump()))
        return value

    @classmethod
    def get(cls, key) -> Optional[OpenAISessionState]:
        result = cls._redis_client.get(key)
        if result:
            return OpenAISessionState(**json.loads(result))
        return None

    @classmethod
    def get_dict(cls, key) -> Optional[Dict]:
        result = cls._redis_client.get(key)
        if result:
            return json.loads(result)
        return None

    @classmethod
    def remove(cls, key) -> None:
        cls._redis_client.delete(key)


def _resolve_path(session: OpenAISessionState, path: str) -> Any:
    """
    Resolve a simple field path on the session.

    Supports:
        "vector_store_id"    -> session.vector_store_id
        "results[2]"         -> session.results[2]  (or None if out of range)
    """
    if "[" in path and path.endswith("]"):
        attr, _, rest = path.partition("[")
        index = int(rest[:-1])
        seq = getattr(session, attr, None)
        if seq is None or index >= len(seq):
            return None
        return seq[index]
    return getattr(session, path, None)


def wait_for_session_field(
    session_id: str,
    path: str,
    timeout: int = 120,
) -> Optional[Any]:
    """
    Poll the session record until the field at `path` is non-None,
    OR session.error_message gets set, OR timeout elapses.

    Returns:
        The field value on success.
        None on timeout, on error_message being set, or on missing session.
        Caller is responsible for distinguishing the None reasons by inspecting
        session.error_message.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        session = FileSearchSession.get(session_id)
        if session is None:
            return None

        if session.error_message:
            return None

        val = _resolve_path(session, path)
        if val is not None:
            return val

        time.sleep(POLL_INTERVAL_SECS)

    return None  # timeout


def session_error(session_id: str) -> Optional[str]:
    """Read session.error_message — used by the Celery task on wait timeouts."""
    session = FileSearchSession.get(session_id)
    return session.error_message if session else None
