from typing import Optional
import logging

from celery import shared_task

from src.file_search.openai_assistant import OpenAIFileAssistant
from src.utils.custom_webhook import CustomWebhook, WebhookConfig


logger = logging.getLogger()


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
        logger.error(err)
        raise Exception(str(err))


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
        logger.error(err)
