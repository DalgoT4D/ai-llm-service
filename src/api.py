import os
import uuid
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, Form
from celery.result import AsyncResult, states
from config.constants import TMP_UPLOAD_DIR_NAME

from src.celerytasks.file_search_tasks import close_file_search_session, query_file
from src.celerytasks.vanna_rag_tasks import train_vanna_on_warehouse, ask_vanna_rag
from src.file_search.openai_assistant import SessionStatusEnum
from src.file_search.session import FileSearchSession, OpenAISessionState
from src.file_search.schemas import FileQueryRequest
from src.vanna.schemas import TrainVannaRequest, AskVannaRequest

router = APIRouter()

logger = logging.getLogger()


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
        # uploading the file to the tmp directory under a session_id
        file_dir = Path(f"{TMP_UPLOAD_DIR_NAME}/{session.id}")
        file_dir.mkdir(parents=True, exist_ok=True)
        fpath = file_dir / file.filename
        with open(fpath, "wb") as buffer:
            buffer.write(file.file.read())

        # update the session
        session.local_fpaths.append(str(fpath))
        session = FileSearchSession.set(session.id, session)

        logger.info("File uploaded successfully")

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
        "error": (
            str(task_result.info)
            if task_result.info and task_result.status != states.SUCCESS
            else None
        ),
    }
    return result


########################### vanna rag related ###########################


@router.post("/vanna/train")
async def post_train_vanna(payload: TrainVannaRequest):
    """Train the vanna RAG against a warehouse for a defined training plan"""
    task = train_vanna_on_warehouse.apply_async(
        kwargs={
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "pg_vector_creds": payload.pg_vector_creds.model_dump(),
            "warehouse_creds": payload.warehouse_creds,
            "training_sql": payload.training_sql,
            "reset": payload.reset,
            "warehouse_type": payload.warehouse_type.value,
        }
    )
    return {"task_id": task.id}


@router.post("/vanna/train/check")
def post_train_vanna_health_check(task_id):
    """Checks if the embeddings are generated or not for the warehouse"""
    return 1


@router.post("/vanna/ask")
async def post_generate_sql(payload: AskVannaRequest):
    """Run the question against the trained vanna RAG to generate a sql query"""
    task = ask_vanna_rag.apply_async(
        kwargs={
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "pg_vector_creds": payload.pg_vector_creds.model_dump(),
            "warehouse_creds": payload.warehouse_creds,
            "warehouse_type": payload.warehouse_type.value,
            "user_prompt": payload.user_prompt,
        }
    )
    return {"task_id": task.id}
