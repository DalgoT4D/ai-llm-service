import os
import logging
import time
from pathlib import Path
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, UploadFile
from celery import shared_task
from celery.result import AsyncResult
from config.constants import TMP_UPLOAD_DIR_NAME


from src.file_search.openai_assistant import OpenAIFileAssistant


router = APIRouter()

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
    file_path: str,
    openai_key: str,
    assistant_prompt: str,
    queries: list[str],
):
    results = []
    try:
        with OpenAIFileAssistant(
            openai_key,
            file_path,
            assistant_prompt,
        ) as fa:
            for i, prompt in enumerate(queries):
                logger.info("%s: %s", i, prompt)
                response = fa.query(prompt)
                logger.info(response)
                results.append(response)

        return results
    except Exception as err:
        logger.error(err)
        raise Exception(
            f"something went wrong generating results in task {self.task_id}"
        )


class FileQueryRequest(BaseModel):
    file_path: str
    assistant_prompt: str
    queries: list[str]


@router.post("/file/query")
async def post_query_file(payload: FileQueryRequest):
    try:
        logger.info("Inside text summarization route")
        task = query_file.apply_async(
            kwargs={
                "file_path": payload.file_path,
                "openai_key": os.getenv("OPENAI_API_KEY"),
                "assistant_prompt": payload.assistant_prompt,
                "queries": payload.queries,
            }
        )
        return {"task_id": task.id}
    except Exception as err:
        logger.error(err)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/file/upload")
async def post_upload_knowledge_file(file: UploadFile):
    try:
        logger.info("reading file contents")
        if file is None:
            raise HTTPException(status_code=400, detail="No file uploaded")
        file_dir = Path(f"{TMP_UPLOAD_DIR_NAME}/{int(time.time())}")
        file_dir.mkdir(parents=True, exist_ok=True)
        with open(file_dir / file.filename, "wb") as buffer:
            buffer.write(file.file.read())

        # TODO: maybe tokenize here; so that we dont give back the bare path
        return {"file_path": file_dir / file.filename}
    except Exception as err:
        logger.error(err)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/task/{task_id}")
def get_summarize_job(task_id):
    """Any queued task can be queried using this endpoint for results"""
    task_result = AsyncResult(task_id)
    result = {
        "id": task_id,
        "status": task_result.status,
        "result": task_result.result,
    }
    return result
