import os
import uuid
import logging
import time
from typing import Optional
from pathlib import Path
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, UploadFile, Form
from celery import shared_task
from celery.result import AsyncResult
from config.constants import TMP_UPLOAD_DIR_NAME


from src.file_search.openai_assistant import OpenAIFileAssistant
from src.file_search.session import FileSearchSession, OpenAISessionState


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
    openai_key: str,
    assistant_prompt: str,
    queries: list[str],
    session_id: str,
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
        raise Exception(str(err))


class FileQueryRequest(BaseModel):
    queries: list[str]
    assistant_prompt: str = None
    session_id: str


@router.delete("/file/search/session/{session_id}")
async def delete_file_search_session(session_id: str):
    """
    Deletes file search session
    1. Deletes open ai resources (file, assistant, thread)
    2. Deletes the tmp file stored in the service
    3. Deletes the session from redis
    """
    try:
        logger.info("Return the file search session")
        task = close_file_search_session.apply_async(
            kwargs={
                "openai_key": os.getenv("OPENAI_API_KEY"),
                "session_id": session_id,
            }
        )
        return {"task_id": task.id}
    except Exception as err:
        logger.error(err)
        raise HTTPException(status_code=500, detail="Failed to get the session")


@router.post("/file/query")
async def post_query_file(payload: FileQueryRequest):
    if payload.queries is None or len(payload.queries) == 0:
        raise HTTPException(status_code=400, detail="Input query is required")

    session = FileSearchSession.get(payload.session_id)
    logger.info("Session: %s", session)

    if not payload.session_id or not session:
        raise HTTPException(status_code=400, detail="Invalid session")

    task = query_file.apply_async(
        kwargs={
            "openai_key": os.getenv("OPENAI_API_KEY"),
            "assistant_prompt": payload.assistant_prompt,
            "queries": payload.queries,
            "session_id": session.id,
        }
    )
    return {"task_id": task.id, "session_id": session.id}


@router.post("/file/upload")
async def post_upload_knowledge_file(file: UploadFile, session_id: str = Form(None)):
    """
    - Upload the document to query on.
    - Starts a session for the file search. Can upload multiple files to the same session.
    - All subsequent queries will be made via this session.
    - Session becomes locked once the first query is made. No more files can be uploaded.
    """

    logger.info(f"Session id requested {session_id}")
    session = None
    if session_id:
        logger.info("Fetching the current session")
        session: OpenAISessionState = FileSearchSession.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

    if not session:
        logger.info("Creating a new session")
        session = OpenAISessionState(
            id=str(uuid.uuid4()),
            local_fpaths=[],
        )

    if session.status == "locked":
        raise HTTPException(
            status_code=400, detail="Session is locked, no more files can be uploaded"
        )

    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        logger.info("reading file contents")
        # uploading the file to the tmp directory under a session_id
        file_dir = Path(f"{TMP_UPLOAD_DIR_NAME}/{session.id}")
        file_dir.mkdir(parents=True, exist_ok=True)
        fpath = file_dir / file.filename
        with open(fpath, "wb") as buffer:
            buffer.write(file.file.read())

        # update the session
        session.local_fpaths.append(str(fpath))
        session = FileSearchSession.set(session.id, session)

        return {"file_path": str(fpath), "session_id": session.id}
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
        "error": str(task_result.info) if task_result.info else None,
    }
    return result
