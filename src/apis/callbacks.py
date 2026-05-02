"""
Inbound webhook receivers from kaapi.

Two endpoints, one per event type:
  - /api/v1/callbacks/collection-ready  → vector store is built
  - /api/v1/callbacks/llm-answer        → an LLM query has answered

Both handlers do exactly two things: validate auth + mutate the session.
No outbound HTTP calls, no business logic. The celery task is the
orchestrator; it polls the session and acts on what these handlers wrote.

Mounted outside the existing authenticate_user dependency in main.py
because kaapi authenticates via its own shared secret, not our client API key.
"""

import os
import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query

from src.file_search.session import FileSearchSession

logger = logging.getLogger()

router = APIRouter()

WEBHOOK_SECRET = os.getenv("KAAPI_WEBHOOK_SECRET", "")


def _verify_kaapi_secret(authorization: Optional[str]) -> None:
    """
    Reject the request if the Authorization header doesn't match the shared secret.

    NOTE: header name/scheme is "Authorization: Bearer <secret>" by default. If
    kaapi uses a different header (e.g. X-Kaapi-Signature), update both the
    parameter binding in the routes below and this function. The exact scheme
    is finalised at deployment time per the migration spec section 5.1.
    """
    if not WEBHOOK_SECRET:
        # No secret configured — fail closed in production-likely envs.
        logger.error("KAAPI_WEBHOOK_SECRET is not configured; rejecting callback")
        raise HTTPException(status_code=401, detail="Webhook secret not configured")

    expected = f"Bearer {WEBHOOK_SECRET}"
    if authorization != expected:
        logger.warning("Rejecting kaapi callback with bad auth header")
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.post("/collection-ready")
def cb_collection_ready(
    payload: dict,
    session_id: str = Query(...),
    authorization: Optional[str] = Header(None),
):
    """
    Kaapi reports that vector store creation has finished (success or failure).
    Body shape mirrors GET /api/v1/collections/jobs/{job_id}.
    """
    _verify_kaapi_secret(authorization)

    session = FileSearchSession.get(session_id)
    if not session:
        logger.warning("collection-ready callback for unknown session %s", session_id)
        return {"status": "received", "response_id": session_id, "message": "no session"}

    data = payload.get("data") or {}
    if data.get("status") == "FAILED":
        session.error_message = data.get("error_message") or "collection failed"
    else:
        collection = data.get("collection") or {}
        session.vector_store_id = collection.get("knowledge_base_id")
        session.collection_id = collection.get("id")

    FileSearchSession.set(session_id, session)
    return {"status": "received", "response_id": session_id, "message": "ok"}


@router.post("/llm-answer")
def cb_llm_answer(
    payload: dict,
    session_id: str = Query(...),
    query_index: int = Query(...),
    authorization: Optional[str] = Header(None),
):
    """
    Kaapi reports the answer for one /llm/call request.
    Body shape mirrors GET /api/v1/llm/call/{job_id} when complete.
    """
    _verify_kaapi_secret(authorization)

    session = FileSearchSession.get(session_id)
    if not session:
        logger.warning("llm-answer callback for unknown session %s", session_id)
        return {"status": "received", "response_id": session_id, "message": "no session"}

    if payload.get("success") is False:
        session.error_message = (
            f"query {query_index} failed: {payload.get('error') or 'unknown'}"
        )
    else:
        try:
            response = payload["data"]["llm_response"]["response"]
            answer_text = response["output"]["text"]
            conv_id = response.get("conversation_id")
        except (KeyError, TypeError) as err:
            logger.error("malformed llm-answer payload for %s: %s", session_id, err)
            session.error_message = f"query {query_index} returned malformed payload"
            FileSearchSession.set(session_id, session)
            return {"status": "received", "response_id": session_id, "message": "ok"}

        # Defensive: pad results so query_index is always in range
        while len(session.results) <= query_index:
            session.results.append(None)
        session.results[query_index] = answer_text

        # First writer wins for conversation_id (subsequent writes carry the same value)
        if session.conversation_id is None and conv_id:
            session.conversation_id = conv_id

    FileSearchSession.set(session_id, session)
    return {"status": "received", "response_id": session_id, "message": "ok"}
