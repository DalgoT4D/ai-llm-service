import os
import tempfile
import json
from enum import Enum

from vanna.openai import OpenAI_Chat
from vanna.pgvector import PG_VectorStore

from langchain_openai import OpenAIEmbeddings


class WarehouseType(str, Enum):
    """
    warehouse types available that vanna model can work with
    """

    POSTGRES = "postgres"
    BIGQUERY = "bigquery"


class CustomVannaClient(PG_VectorStore, OpenAI_Chat):
    """
    Vanna client with pgvector as its backend and openai as the service provider
    All RAG related calls to talk vanna model will be made via this client
    """

    def __init__(
        self,
        openai_api_key: str,
        pg_vector_creds: dict,
        openai_model: str = "gpt-4o-mini",
        initial_prompt: str = None,
    ):
        pg_vector_db_keys = pg_vector_creds.keys()
        if not all(
            key in pg_vector_db_keys
            for key in ["username", "password", "server", "port", "database"]
        ):
            raise ValueError("Invalid pg vector creds")

        PG_VectorStore.__init__(
            self,
            config={
                "connection_string": "postgresql+psycopg://{username}:{password}@{server}:{port}/{database}".format(
                    **pg_vector_creds
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
        self, pg_vector_creds: dict, warehouse_creds: dict, warehouse_type: str
    ):
        self.vanna = CustomVannaClient(
            openai_api_key=os.getenv("OPENAI_API_KEY"), pg_vector_creds=pg_vector_creds
        )
        if warehouse_type == WarehouseType.POSTGRES:
            required_creds = {
                "host": warehouse_creds["host"],
                "port": warehouse_creds["port"],
                "dbname": warehouse_creds["database"],
                "user": warehouse_creds["username"],
                "password": warehouse_creds["password"],
            }

            self.vanna.connect_to_postgres(**required_creds)
        elif warehouse_type == WarehouseType.BIGQUERY:
            cred_file_path = None
            with tempfile.NamedTemporaryFile(
                delete=False, mode="w", suffix=".json"
            ) as temp_file:
                json.dump(warehouse_creds, temp_file, indent=4)
                cred_file_path = temp_file.name

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
        self.vanna.remove_training_data()
        return True
