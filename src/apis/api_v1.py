import os
import uuid
import logging
from typing import Optional
from pathlib import Path
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, UploadFile, Form


from src.file_search.openai_assistant import SessionStatusEnum
from src.file_search.session import FileSearchSession, OpenAISessionState
from src.custom_webhook import WebhookConfig
from src.services import ai_platform_src
from src.utils.celery_tasks import query_file_v1, close_file_search_session_v1


router = APIRouter()

logger = logging.getLogger()


class FileQueryRequest(BaseModel):
    queries: list[str]
    assistant_prompt: str = None
    session_id: str
    webhook_config: Optional[WebhookConfig] = None


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
        task = close_file_search_session_v1.apply_async(
            kwargs={
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

    logger.info("Starting the file query task")

    task = query_file_v1.apply_async(
        kwargs={
            "assistant_prompt": payload.assistant_prompt,
            "queries": payload.queries,
            "session_id": session.id,
            "webhook_config": (
                payload.webhook_config.model_dump() if payload.webhook_config else None
            ),
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

    if session.status == SessionStatusEnum.locked:
        raise HTTPException(
            status_code=400, detail="Session is locked, no more files can be uploaded"
        )

    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        logger.info("reading file contents")
        # uploading the file
        document_id = ai_platform_src.upload_document(file)

        session.document_ids.append(document_id)

        # update the session
        session = FileSearchSession.set(session.id, session)

        logger.info("File uploaded successfully")

        return {"file_path": document_id, "session_id": session.id}
    except Exception as err:
        logger.error(err)
        raise HTTPException(status_code=500, detail="Internal Server Error")
