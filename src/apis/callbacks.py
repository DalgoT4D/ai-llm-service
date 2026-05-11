"""
Inbound webhook receivers from kaapi.

Two endpoints, one per event type:
  - /api/v1/callbacks/collection-ready  → vector store is built
  - /api/v1/callbacks/llm-answer        → an LLM query has answered

Both handlers do exactly two things: validate auth + mutate the session.
No outbound HTTP calls, no business logic. The celery task is the
orchestrator; it polls the session and acts on what these handlers wrote.

Mounted outside the existing authenticate_user dependency in main.py
because kaapi authenticates via HMAC-SHA256 over the request body, not
our client API key.

Auth scheme — 
    Headers:
        X-Webhook-Signature: <hex HMAC-SHA256>
        X-Webhook-Timestamp: <unix milliseconds>
    Signing string:
        f"{timestamp_ms}.".encode() + raw_body_bytes
    Algorithm:
        hmac.new(secret, signing_string, sha256).hexdigest()
"""

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from src.file_search.session import FileSearchSession

logger = logging.getLogger()

router = APIRouter()

WEBHOOK_SECRET = os.getenv("KAAPI_WEBHOOK_SECRET", "")

# Reject callbacks whose timestamp is more than this far from "now".
_MAX_WEBHOOK_AGE_SECS = 300


def _verify_kaapi_signature(
    raw_body: bytes,
    signature: Optional[str],
    timestamp_ms: Optional[str],
) -> None:
    """
    Verify kaapi's HMAC-SHA256 webhook signature. Raises 401 on mismatch.
    Kaapi creates a X-Webhook-Signature using current timestamp + body + secret and then we also create the same signature and compare.
    """
    if not WEBHOOK_SECRET:
        logger.error("KAAPI_WEBHOOK_SECRET not configured; rejecting callback")
        raise HTTPException(status_code=401, detail="Webhook secret not configured")

    if not signature or not timestamp_ms:
        logger.warning("Missing X-Webhook-Signature or X-Webhook-Timestamp header")
        raise HTTPException(status_code=401, detail="Missing signature headers")

    try:
        ts = int(timestamp_ms)
    except ValueError:
        raise HTTPException(status_code=401, detail="Bad timestamp")

    now_ms = int(time.time() * 1000)
    if abs(now_ms - ts) > _MAX_WEBHOOK_AGE_SECS * 1000:
        logger.warning("Webhook timestamp outside replay window: ts=%s now=%s",
                       ts, now_ms)
        raise HTTPException(status_code=401, detail="Stale timestamp")

    signing_string = f"{ts}.".encode() + raw_body
    expected = hmac.new(
        WEBHOOK_SECRET.encode(), signing_string, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, signature):
        logger.warning("Bad webhook signature for ts=%s", ts)
        raise HTTPException(status_code=401, detail="Unauthorized")


async def _verify_and_parse(request: Request) -> dict:
    raw_body = await request.body()
    _verify_kaapi_signature(
        raw_body,
        request.headers.get("x-webhook-signature"),
        request.headers.get("x-webhook-timestamp"),
    )
    if not raw_body:
        return {}
    try:
        return json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Bad JSON")


@router.post("/collection-ready")
async def cb_collection_ready(
    request: Request,
    session_id: str = Query(...),
):
    """
    Kaapi reports that vector store creation has finished (success or failure).
    Body shape mirrors our older GET /api/v1/collections/jobs/{job_id}.
    """
    payload = await _verify_and_parse(request)

    session = FileSearchSession.get(session_id)
    if not session:
        logger.warning("collection-ready callback for unknown session %s", session_id)
        return {"status": "received", "response_id": session_id, "message": "no session"}

    data = payload.get("data") or {}
    if data.get("status") == "FAILED":
        session.error_message = data.get("error_message") or "collection failed"
    else:
        try:
            collection = data["collection"]
            session.vector_store_id = collection["knowledge_base_id"]
        except (KeyError, TypeError) as err:
            logger.error("malformed collection-ready payload for %s: %s", session_id, err)
            session.error_message = "collection creation returned malformed payload"

    FileSearchSession.set(session_id, session)
    return {"status": "received", "response_id": session_id, "message": "ok"}


@router.post("/llm-answer")
async def cb_llm_answer(
    request: Request,
    session_id: str = Query(...),
    query_index: int = Query(...),
):
    """
    Kaapi reports the answer for one /llm/call request.
    Real shape per kaapi OpenAPI 0.5.0:
        data.response.output.{type, content.{format, value}}
    """
    payload = await _verify_and_parse(request)

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
            response = payload["data"]["response"]
            answer_text = response["output"]["content"]["value"]
            conv_id = response.get("conversation_id")
        except (KeyError, TypeError) as err:
            logger.error("malformed llm-answer payload for %s: %s", session_id, err)
            session.error_message = f"query {query_index} returned malformed payload"
            FileSearchSession.set(session_id, session)
            return {"status": "received", "response_id": session_id, "message": "ok"}

        session.results[query_index] = answer_text

        if session.conversation_id is None and conv_id:
            session.conversation_id = conv_id

    FileSearchSession.set(session_id, session)
    return {"status": "received", "response_id": session_id, "message": "ok"}
