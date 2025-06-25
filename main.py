# Load .env as early as possible, before any imports that may use env vars
from dotenv import load_dotenv

load_dotenv()

import os
import uvicorn
from pathlib import Path
import logging
from fastapi import FastAPI, Depends, Security, status, HTTPException
from fastapi.security import (
    HTTPBearer,
    APIKeyHeader,
)
from celery import Celery
from logging.config import dictConfig

from src.apis.api import router as text_summarization_router
from src.apis.api_v1 import router as text_summarization_router_v1
from config.celery_config import CeleryConfig
from config.constants import TMP_UPLOAD_DIR_NAME, LOGS_DIR_NAME

log_dir = Path(__file__).resolve().parent / LOGS_DIR_NAME
log_dir.mkdir(parents=True, exist_ok=True)

tmp_upload_dir = Path(__file__).resolve().parent / TMP_UPLOAD_DIR_NAME
tmp_upload_dir.mkdir(parents=True, exist_ok=True)


# logging configuration
class CustomFormatter(logging.Formatter):
    def format(self, record):
        cwd = os.getcwd()
        abs_path = record.pathname
        rel_path = os.path.relpath(abs_path, cwd)
        record.pathname = rel_path
        return super().format(record)


dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "()": CustomFormatter,
                "format": "[%(asctime)s] %(levelname)s in %(pathname)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S %Z",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/app.log",
                "maxBytes": 1048576,  # 1MB
                "backupCount": 5,
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)


app = FastAPI()

security = HTTPBearer()

api_key_header = APIKeyHeader(name="Authorization")


async def authenticate_user(api_key_header: str = Security(api_key_header)):
    if api_key_header != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )
    return {}


celery = Celery(
    "t4d-ai-llm",
)
celery.config_from_object(CeleryConfig, namespace="CELERY")
celery.autodiscover_tasks(
    [
        "src.apis",
    ]
)

# routes
app.include_router(
    text_summarization_router, prefix="/api", dependencies=[Depends(authenticate_user)]
)
app.include_router(
    text_summarization_router_v1,
    prefix="/api/v1",
    dependencies=[Depends(authenticate_user)],
)


@app.get("/health")
def health_check():
    """
    Health check endpoint to verify if the service is running.
    """
    return {"status": "ok"}


# home route
@app.get("/api")
async def home(auth_user: dict = Depends(authenticate_user)):
    return {"message": "Welcome to the T4D's AI/LLM service"}


if __name__ == "__main__":
    uvicorn.run("main:app", port=7001, reload=True, reload_dirs=["src", "config"])
