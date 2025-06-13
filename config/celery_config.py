import os
from kombu import Queue


def route_task(name, args, kwargs, options, task=None, **kw):
    if ":" in name:
        queue, _ = name.split(":")
        return {"queue": queue}
    return {"queue": "llm"}


class CeleryConfig:
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    CELERY_TASK_DEFAULT_QUEUE = "llm"
    CELERY_TASK_QUEUES: list = (
        # default queue
        Queue("llm"),
    )
    CELERY_TASK_ROUTES = (route_task,)
    broker_connection_retry_on_startup = True

    # Instead, use autodiscover_tasks in your Celery app initialization (main.py or celery.py):
    # celery.autodiscover_tasks(['src.apis'])
