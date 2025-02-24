from enum import Enum
from typing import Optional
from pydantic import BaseModel


class WarehouseType(str, Enum):
    """
    warehouse types available that vanna model can work with
    """

    POSTGRES = "postgres"
    BIGQUERY = "bigquery"


class PgVectorCreds(BaseModel):
    """Pg Vector Creds where the embeddings for the RAG will be stored"""

    username: str
    password: str
    host: str
    port: int
    database: str


class BaseVannaWarehouseConfig(BaseModel):
    """Base model for vanna related stuff"""

    pg_vector_creds: PgVectorCreds
    warehouse_creds: dict
    warehouse_type: WarehouseType


class TrainVannaRequest(BaseVannaWarehouseConfig):
    """Payload to train vanna model against a warehouse"""

    training_sql: str
    reset: bool = True


class AskVannaRequest(BaseVannaWarehouseConfig):
    """Payload to ask vanna for sql corresponding to a user prompt"""

    user_prompt: str
