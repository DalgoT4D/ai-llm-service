import os
import uvicorn
from pathlib import Path
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from celery import Celery
from starlette.middleware.base import BaseHTTPMiddleware
from logging.config import dictConfig
from dotenv import load_dotenv

from src.api import router as text_summarization_router
from config.celery_config import CeleryConfig

log_dir = Path(__file__).resolve().parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)


load_dotenv()


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


class CustomAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Add your custom logic here
        if request.headers.get("authorization") != os.getenv("API_KEY"):
            logging.info("hi there")
            return JSONResponse(status_code=401, content={"message": "Unauthorized"})
        return await call_next(request)


app = FastAPI()

# celery
celery = Celery(
    "t4d-ai-llm",
)
celery.config_from_object(CeleryConfig, namespace="CELERY")

# middleware
app.add_middleware(CustomAuthMiddleware)

# routes
app.include_router(text_summarization_router, prefix="/api")


# home route
@app.get("/api")
async def home():
    return {"message": "Welcome to the T4D's AI/LLM service"}


if __name__ == "__main__":
    uvicorn.run("main:app", port=7001, reload=True)
