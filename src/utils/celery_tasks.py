import logging
import traceback
from typing import Optional

from celery import shared_task
from fastapi import HTTPException

from src.custom_webhook import CustomWebhook, WebhookConfig
from src.file_search.session import FileSearchSession
from src.file_search.openai_assistant import OpenAIFileAssistant
from src.services import ai_platform_src

logger = logging.getLogger()


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=5,  # tasks will retry after 5, 10, 15... seconds
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
    try:
        # get the session
        session = FileSearchSession.get(session_id)
        logger.info("Session: %s", session)

        # create collection
        collection_id = ai_platform_src.create_collection(
            ai_platform_src.CollectionCreatePayload(
                instructions=assistant_prompt,
                documents=session.document_ids,
                model="gpt-4o",
                temperature=0.000001,
                batch_size=1,
            )
        )

        # wait till the collection is created
        collection: dict = ai_platform_src.poll_collection_creation(
            collection_id=collection_id
        )
        if not collection:
            logger.error("Collection creation failed")
            raise HTTPException(
                status_code=500,
                detail="Collection creation failed; something went wrong",
            )
        logger.info("Collection created successfully")

        results = []

        thread_id = None
        for i, prompt in enumerate(queries):
            logger.info("Starting query %s: %s", i, prompt)
            # start a thread with the query
            thread_id = ai_platform_src.create_and_start_thread(
                ai_platform_src.CreateAndStartThreadPayload(
                    question=prompt,
                    assistant_id=collection["llm_service_id"],
                    remove_citation=True,
                    thread_id=thread_id,  # Use the thread_id from the previous iteration
                )
            )
            logger.info("Thread created successfully with ID: %s", thread_id)

            response = ai_platform_src.poll_thread_result(thread_id=thread_id)

            results.append(response)

        if webhook_config:
            webhook = CustomWebhook(WebhookConfig(**webhook_config))
            logger.info(
                f"Posting results to the webhook configured at {webhook.config.endpoint}"
            )
            res = webhook.post_result({"results": results, "session_id": session_id})
            logger.info(f"Results posted to the webhook with res: {str(res)}")

        return {"result": results, "session_id": session_id}
    except Exception as err:
        logger.error(traceback.format_exc())  # Log the full traceback
        raise Exception(traceback.format_exc())  # Raise with full traceback


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=5,  # tasks will retry after 5, 10, 15... seconds
    retry_kwargs={"max_retries": 3},
    name="close_file_search_session_v1",
    logger=logging.getLogger(),
)
def close_file_search_session_v1(self, session_id: str):
    try:
        session = FileSearchSession.get(session_id)
        logger.info("Session: %s", session)

        if not session:
            raise Exception("Invalid session")

        for document_id in session.document_ids:
            logger.info(f"Deleting document {document_id}")
            ai_platform_src.delete_document(document_id)
    except Exception as err:
        logger.error(traceback.format_exc())
        raise Exception(traceback.format_exc())


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=5,  # tasks will retry after 5, 10, 15... seconds
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
        logger.error(traceback.format_exc())  # Log the full traceback
        raise Exception(traceback.format_exc())  # Raise with full traceback


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=5,  # tasks will retry after 5, 10, 15... seconds
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
