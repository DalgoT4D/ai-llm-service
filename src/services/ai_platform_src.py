import os
import logging
from typing import Optional, Any
from pydantic import BaseModel

from fastapi import UploadFile, HTTPException
from src.utils.http_helper import http_post, http_delete

logger = logging.getLogger()

API_KEY = os.getenv("AI_PLATFORM_API_KEY")
BASE_URI = os.getenv("AI_PLATFORM_BASE_URI")
TIMEOUT = int(os.getenv("AI_PLATFORM_REQUEST_TIMEOUT_SECS", 120))
PUBLIC_BASE = os.getenv("AI_LLM_SERVICE_PUBLIC_BASE", "")
HEADERS = {"x-api-key": f"ApiKey {API_KEY}"}

if not BASE_URI:
    raise HTTPException(
        status_code=500,
        detail="AI Platform base URI is not configured. Please set the AI_PLATFORM_BASE_URI environment variable.",
    )


# --------- request model ---------

class CollectionCreatePayload(BaseModel):
    """
    Vector-store-only collection creation payload.

    IMPORTANT: do NOT add `model`, `instructions`, or `temperature` to this model.
    Their presence in the request body triggers kaapi to create an OpenAI Assistant
    in addition to a vector store. We want vector store only. `extra = "forbid"`
    enforces this at the Pydantic level.
    """

    documents: list[str]
    provider: str = "openai"
    callback_url: str
    name: Optional[str] = None
    description: Optional[str] = None

    class Config:
        extra = "forbid"


# --------- callback URL builders ---------

def collection_callback_url(session_id: str) -> str:
    return f"{PUBLIC_BASE}/api/v1/callbacks/collection-ready?session_id={session_id}"


def llm_answer_callback_url(session_id: str, query_index: int) -> str:
    return (
        f"{PUBLIC_BASE}/api/v1/callbacks/llm-answer"
        f"?session_id={session_id}&query_index={query_index}"
    )


# --------- outbound calls to kaapi ---------

def upload_document(file: UploadFile) -> str:
    """
    Upload a document to kaapi. Returns the document_id.

    UNCHANGED from previous flow — kaapi's /documents/ endpoint hasn't changed.
    """
    upload_url = f"{BASE_URI}/documents/"
    content_type = file.content_type or "application/octet-stream"
    files = {"src": (file.filename, file.file, content_type)}
    res = http_post(upload_url, files=files, headers=HEADERS)

    if not res or not res.get("data") or not res["data"].get("id"):
        raise HTTPException(
            status_code=500,
            detail=f"Invalid response from document upload API: {res}",
        )
    return res["data"]["id"]


def create_collection(payload: CollectionCreatePayload) -> str:
    """
    Fire vector-store-only collection creation. Returns kaapi's job_id.

    The job_id is logged but not used downstream. We wait for the result via
    the callback at `payload.callback_url`, not by polling.
    """
    create_collection_url = f"{BASE_URI}/collections/"
    res = http_post(create_collection_url, json=payload.model_dump(), headers=HEADERS)

    if not res or not res.get("data") or not res["data"].get("job_id"):
        raise HTTPException(
            status_code=500,
            detail=f"Invalid response from collection create API: {res}",
        )

    job_id = res["data"]["job_id"]
    logger.info("Collection job %s queued at kaapi (callback to %s)",
                job_id, payload.callback_url)
    return job_id


def llm_call(
    *,
    query: str,
    vector_store_id: str,
    instructions: str,
    callback_url: str,
    conversation_id: Optional[str] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 1.0,
) -> str:
    """
    Fire a single LLM call against the vector store. Returns kaapi's job_id.
    The actual answer arrives later via `callback_url`.

    Replaces the deprecated /threads/start + /threads/result/{id} pair.
    """
    if conversation_id:
        conversation: dict[str, Any] = {"id": conversation_id, "auto_create": False}
    else:
        conversation = {"auto_create": True}

    body = {
        "query": {
            "input": query,
            "conversation": conversation,
        },
        "config": {
            "blob": {
                "completion": {
                    "provider": "openai",
                    "type": "text",
                    "params": {
                        "model": model,
                        "instructions": instructions,
                        "temperature": temperature,
                        "knowledge_base_ids": [vector_store_id],
                    },
                }
            }
        },
        "callback_url": callback_url,
        "include_provider_raw_response": False,
    }

    llm_call_url = f"{BASE_URI}/llm/call"
    res = http_post(llm_call_url, json=body, headers=HEADERS)

    if not res or not res.get("data") or not res["data"].get("job_id"):
        raise HTTPException(
            status_code=500,
            detail=f"Invalid response from /llm/call: {res}",
        )

    job_id = res["data"]["job_id"]
    logger.info("LLM call %s queued at kaapi (callback to %s)", job_id, callback_url)
    return job_id


def delete_document(document_id: str) -> bool:
    """Delete a document from kaapi. UNCHANGED."""
    delete_url = f"{BASE_URI}/documents/{document_id}"
    res = http_delete(delete_url, headers=HEADERS)

    if not res.get("success", False):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document {document_id}: {res.get('error')}",
        )
    return True


def delete_collection(collection_id: str) -> bool:
    """Delete a collection (vector store) from kaapi. NEW for cleanup."""
    delete_url = f"{BASE_URI}/collections/{collection_id}"
    res = http_delete(delete_url, headers=HEADERS)

    if not res.get("success", False):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete collection {collection_id}: {res.get('error')}",
        )
    return True
