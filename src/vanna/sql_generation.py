import os
import tempfile
import json
from enum import Enum
import logging
from sqlalchemy import create_engine, text

from openai import OpenAI
from vanna.openai import OpenAI_Chat
from vanna.pgvector import PG_VectorStore
from langchain_openai import OpenAIEmbeddings

from src.vanna.schemas import PgVectorCreds, WarehouseType


logger = logging.getLogger()


class CustomPG_VectorStore(PG_VectorStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def remove_all_training_data(self, **kwargs):
        engine = create_engine(self.connection_string)

        delete_statement = text(
            """
            DELETE FROM langchain_pg_embedding
            """
        )

        with engine.connect() as connection:
            with connection.begin() as transaction:
                try:
                    result = connection.execute(delete_statement)
                    transaction.commit()
                    return result.rowcount > 0
                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    transaction.rollback()
                    return False

    def cnt_of_embeddings(self):
        engine = create_engine(self.connection_string)

        stmt = text(
            """
            SELECT count(*) FROM langchain_pg_embedding
            """
        )

        with engine.connect() as connection:
            with connection.begin() as transaction:
                try:
                    result = connection.execute(stmt)
                    transaction.commit()
                    return result.fetchone()[0]
                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    transaction.rollback()
                    return False


class CustomVannaClient(CustomPG_VectorStore, OpenAI_Chat):
    """
    Vanna client with pgvector as its backend and openai as the service provider
    All RAG related calls to talk vanna model will be made via this client
    """

    def __init__(
        self,
        openai_api_key: str,
        pg_vector_creds: PgVectorCreds,
        openai_model: str = "gpt-4o-mini",
        initial_prompt: str = None,
    ):
        CustomPG_VectorStore.__init__(
            self,
            config={
                "connection_string": "postgresql+psycopg://{username}:{password}@{host}:{port}/{database}".format(
                    **pg_vector_creds.model_dump()
                ),
                "embedding_function": OpenAIEmbeddings(),
            },
        )

        OpenAI_Chat.__init__(
            self,
            config={
                "api_key": openai_api_key,
                "model": openai_model,
                "initial_prompt": initial_prompt,
            },
        )


class SqlGeneration:
    def __init__(
        self,
        openai_api_key: str,
        pg_vector_creds: PgVectorCreds,
        warehouse_creds: dict,
        warehouse_type: str,
    ):
        os.environ["OPENAI_API_KEY"] = openai_api_key

        if warehouse_type == WarehouseType.POSTGRES:
            required_creds = {
                "host": warehouse_creds["host"],
                "port": warehouse_creds["port"],
                "dbname": warehouse_creds["database"],
                "user": warehouse_creds["username"],
                "password": warehouse_creds["password"],
            }

            self.vanna = CustomVannaClient(
                openai_api_key=openai_api_key,
                pg_vector_creds=pg_vector_creds,
                initial_prompt="please include schema names in queries",
            )
            self.vanna.connect_to_postgres(**required_creds)
        elif warehouse_type == WarehouseType.BIGQUERY:
            cred_file_path = None
            with tempfile.NamedTemporaryFile(
                delete=False, mode="w", suffix=".json"
            ) as temp_file:
                json.dump(warehouse_creds, temp_file, indent=4)
                cred_file_path = temp_file.name

            self.vanna = CustomVannaClient(
                openai_api_key=openai_api_key,
                pg_vector_creds=pg_vector_creds,
                initial_prompt="please include backticks for project names and table names if appropriate",
            )
            self.vanna.connect_to_bigquery(
                project_id=warehouse_creds["project_id"],
                cred_file_path=cred_file_path,
            )
        else:
            raise ValueError("Invalid warehouse type")

    def generate_sql(self, question: str, allow_llm_to_see_data=False):
        return self.vanna.generate_sql(
            question=question, allow_llm_to_see_data=allow_llm_to_see_data
        )

    def is_sql_valid(self, sql: str):
        return self.vanna.is_sql_valid(sql=sql)

    def run_sql(self, sql: str):
        return self.vanna.run_sql(sql=sql)

    def setup_training_plan_and_execute(self, training_sql: str):
        df_information_schema = self.vanna.run_sql(training_sql)
        plan = self.vanna.get_training_plan_generic(df_information_schema)
        self.vanna.train(plan=plan)
        return True

    def remove_training_data(self):
        self.vanna.remove_all_training_data()
        return True

    def is_trained(self) -> bool:
        return self.vanna.cnt_of_embeddings() > 0
