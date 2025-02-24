from typing import Optional
from celery import shared_task
import logging

from src.vanna.schemas import PgVectorCreds
from src.vanna.sql_generation import SqlGeneration


logger = logging.getLogger()


@shared_task(
    bind=True,
    retry_backoff=5,  # tasks will retry after 5, 10, 15... seconds
    retry_kwargs={"max_retries": 1},
    name="train_vanna_on_warehouse",
    logger=logging.getLogger(),
)
def train_vanna_on_warehouse(
    self,
    openai_api_key: str,
    pg_vector_creds: dict,
    warehouse_creds: dict,
    training_sql: str,
    reset: bool,
    warehouse_type: str,
):

    sql_generation_client = SqlGeneration(
        openai_api_key=openai_api_key,
        pg_vector_creds=PgVectorCreds(**pg_vector_creds),
        warehouse_creds=warehouse_creds,
        warehouse_type=warehouse_type,
    )

    if reset:
        sql_generation_client.remove_training_data()
        logger.info("Deleted training data successfully")

    sql_generation_client.setup_training_plan_and_execute(training_sql)

    logger.info(
        f"Completed training successfully with the following plan {training_sql}"
    )

    return True


@shared_task(
    bind=True,
    retry_backoff=5,  # tasks will retry after 5, 10, 15... seconds
    retry_kwargs={"max_retries": 1},
    name="ask_vanna_rag",
    logger=logging.getLogger(),
)
def ask_vanna_rag(
    self,
    openai_api_key: str,
    pg_vector_creds: dict,
    warehouse_creds: dict,
    warehouse_type: str,
    user_prompt: str,
):

    sql_generation_client = SqlGeneration(
        openai_api_key=openai_api_key,
        pg_vector_creds=PgVectorCreds(**pg_vector_creds),
        warehouse_creds=warehouse_creds,
        warehouse_type=warehouse_type,
    )

    logger.info("Starting sql generation")

    sql = sql_generation_client.generate_sql(
        question=user_prompt, allow_llm_to_see_data=False
    )

    logger.info(f"Finished sql generation with result: {sql}")

    if not sql_generation_client.is_sql_valid(sql):
        raise Exception(f"Failed to get a valid sql from llm : {sql}")

    return sql
