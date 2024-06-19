import os
import logging
from fastapi import APIRouter, HTTPException
from celery import shared_task


router = APIRouter()


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
    name="summarize_logs",
)
def summarize_logs(self):
    logging.info("Inside summarize_logs task")

    logging.info("Call openai apis here")


@router.post("/summarize")
async def post_summarize():
    try:
        logging.info("Inside text summarization route")
        task = summarize_logs.apply_async()
        return {"task_id": task.id}
    except Exception as err:
        logging.error(err)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/job/{job_id}")
def get_summarize_job(job_id):
    """Any queued job can be queried using this endpoint for results"""
    return {"message": "sccc"}
