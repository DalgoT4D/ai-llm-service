import logging
import traceback
from typing import Optional

from celery import shared_task

from src.custom_webhook import CustomWebhook, WebhookConfig
from src.file_search.session import (
    FileSearchSession,
    wait_for_session_field,
    session_error,
)
from src.file_search.openai_assistant import OpenAIFileAssistant
from src.services import ai_platform_src

logger = logging.getLogger()


# timeouts in case of any crash
_COLLECTION_TIMEOUT_SECS = 120
_LLM_CALL_TIMEOUT_SECS = 120


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=5,
    retry_kwargs={"max_retries": 0},
    name="query_file_v1",
    logger=logging.getLogger(),
)
def query_file_v1(
    self,
    assistant_prompt: str,
    queries: list[str],
    session_id: str,
    webhook_config: Optional[dict] = None,
):
    """
    Orchestrates the kaapi vector-store + llm/call flow for one /file/query call.

    Flow:
      1. Reset in-flight session fields (queries, results, error_message).
      2. Fire POST /collections/ to kaapi with our callback_url.
      3. Block on session.vector_store_id (filled by /callbacks/collection-ready).
      4. For each query: fire POST /llm/call, block on session.results[i]
         (filled by /callbacks/llm-answer). Reuse conversation_id across calls.
      5. Return session.results.

    DDP_backend keeps polling /api/task/{task_id} until Celery flips status
    to SUCCESS (when this function returns) or FAILURE (when it raises).
    """
    try:
        session = FileSearchSession.get(session_id)
        if not session:
            raise Exception("Invalid session")

        # 1. Reset in-flight fields. Critical: stale results from a previous
        # /file/query call on this session would otherwise look like completed work.
        session.queries = queries
        session.results = [None] * len(queries)
        session.error_message = None
        FileSearchSession.set(session_id, session)

        # 2. Fire collection creation (kaapi will call our webhook when done)
        # returns a job_id but we don't need it now. 
        ai_platform_src.create_collection(
            ai_platform_src.CollectionCreatePayload(
                documents=session.document_ids or [],
                provider="openai",
                callback_url=ai_platform_src.collection_callback_url(session_id),
                name=f"dalgo-session-{session_id}",
            )
        )

        # 3. Wait for vector_store_id to land on the session
        vector_store_id = wait_for_session_field(
            session_id, "vector_store_id", timeout=_COLLECTION_TIMEOUT_SECS
        )
        if vector_store_id is None:
            err = session_error(session_id) or "collection creation timed out"
            raise Exception(err)

        # 4. Fire each query, wait for each answer (sequential, chained via conversation_id)
        conversation_id: Optional[str] = None
        for i, query in enumerate(queries):
            logger.info("Firing LLM call %s/%s for session %s",
                        i + 1, len(queries), session_id)
            ai_platform_src.llm_call(
                query=query,
                vector_store_id=vector_store_id,
                conversation_id=conversation_id, # first time auto_create = true. 
                instructions=assistant_prompt,
                callback_url=ai_platform_src.llm_answer_callback_url(session_id, i),
            )
            answer = wait_for_session_field(
                session_id, f"results[{i}]", timeout=_LLM_CALL_TIMEOUT_SECS
            )
            if answer is None:
                err = session_error(session_id) or f"query {i} timed out"
                raise Exception(err)

            # Capture conversation_id from session after the first answer; reuse for rest
            if conversation_id is None:
                refreshed = FileSearchSession.get(session_id)
                conversation_id = refreshed.conversation_id if refreshed else None

        # 5. Read final results from the session
        final = FileSearchSession.get(session_id)
        results = final.results if final else []

        # 6. Optional client webhook (DDP_backend doesn't use this today)
        if webhook_config:
            webhook = CustomWebhook(WebhookConfig(**webhook_config))
            logger.info("Posting results to webhook %s", webhook.config.endpoint)
            res = webhook.post_result({"results": results, "session_id": session_id})
            logger.info("Webhook post result: %s", str(res))

        return {"result": results, "session_id": session_id}

    except Exception:
        logger.error(traceback.format_exc())
        raise


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=5,
    retry_kwargs={"max_retries": 3},
    name="close_file_search_session_v1",
    logger=logging.getLogger(),
)
def close_file_search_session_v1(self, session_id: str):
    """
    Cleanup at session close:
      1. DELETE the kaapi collection (vector store) if we know its id.
      2. DELETE each kaapi document we uploaded.
      3. Remove the session record from Redis.

    Failures on individual deletes are logged but do not abort the rest —
    we want best-effort cleanup so a stuck remote resource doesn't leak forever.
    """
    try:
        session = FileSearchSession.get(session_id)
        if not session:
            logger.info("close_file_search_session_v1: session %s already gone",
                        session_id)
            return

        if session.collection_id:
            logger.info("Deleting kaapi collection %s", session.collection_id)
            try:
                ai_platform_src.delete_collection(session.collection_id)
            except Exception as err:
                logger.warning("Failed to delete collection %s: %s",
                               session.collection_id, err)

        for document_id in session.document_ids or []:
            logger.info("Deleting kaapi document %s", document_id)
            try:
                ai_platform_src.delete_document(document_id)
            except Exception as err:
                logger.warning("Failed to delete document %s: %s", document_id, err)

        FileSearchSession.remove(session_id)
    except Exception:
        logger.error(traceback.format_exc())
        raise


# ---------- legacy v0 tasks below (UNCHANGED, talk to OpenAI Assistants API directly) ----------


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=5,
    retry_kwargs={"max_retries": 3},
    name="query_file",
    logger=logging.getLogger(),
)
def query_file(
    self,
    openai_key: str,
    assistant_prompt: str,
    queries: list[str],
    session_id: str,
    webhook_config: Optional[dict] = None,
):
    fa = None
    try:
        results = []

        fa = OpenAIFileAssistant(
            openai_key,
            session_id=session_id,
            instructions=assistant_prompt,
        )
        for i, prompt in enumerate(queries):
            logger.info("%s: %s", i, prompt)
            response = fa.query(prompt)
            results.append(response)

        logger.info(f"Results generated in the session {fa.session.id}")

        if webhook_config:
            webhook = CustomWebhook(WebhookConfig(**webhook_config))
            logger.info(
                f"Posting results to the webhook configured at {webhook.config.endpoint}"
            )
            res = webhook.post_result({"results": results, "session_id": fa.session.id})
            logger.info(f"Results posted to the webhook with res: {str(res)}")

        return {"result": results, "session_id": fa.session.id}
    except Exception as err:
        logger.error(traceback.format_exc())
        raise Exception(traceback.format_exc())


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=5,
    retry_kwargs={"max_retries": 3},
    name="close_file_search_session",
    logger=logging.getLogger(),
)
def close_file_search_session(self, openai_key, session_id: str):
    try:
        fa = OpenAIFileAssistant(openai_key, session_id=session_id)
        fa.close()
    except Exception as err:
        logger.error(traceback.format_exc())
        raise Exception(traceback.format_exc())
