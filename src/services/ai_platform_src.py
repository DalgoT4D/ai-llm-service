import os
import logging
from pydantic import BaseModel
import time
from typing import Optional

from fastapi import UploadFile, HTTPException
from src.utils.http_helper import http_post, http_get, http_delete

logger = logging.getLogger()

API_KEY = os.getenv("AI_PLATFORM_API_KEY")
BASE_URI = os.getenv("AI_PLATFORM_BASE_URI")
POLLING_INTERVAL = int(os.getenv("AI_PLATFORM_POLLING_INTERVAL", 5))
TIMEOUT = int(os.getenv("AI_PLATFORM_REQUEST_TIMEOUT_SECS", 120))
HEADERS = {"x-api-key": f"ApiKey {API_KEY}"}

if not BASE_URI:
    raise HTTPException(
        status_code=500,
        detail="AI Platform base URI is not configured. Please set the AI_PLATFORM_BASE_URI environment variable.",
    )


class CollectionCreatePayload(BaseModel):
    instructions: str
    documents: list[str] = []
    model: str = "gpt-4o"
    temperature: float = 0.000001
    batch_size: int = 1

    class Config:
        extra = "allow"


class CreateAndStartThreadPayload(BaseModel):
    question: str
    assistant_id: str
    remove_citation: bool = True
    thread_id: Optional[str] = None
    project_id: int = 1


def upload_document(file: UploadFile) -> str:
    """
    Uploads a document to the external platform.

    Args:
        file: FastAPI UploadFile

    Returns:
        str: ID of the uploaded document.
    """

    upload_url = f"{BASE_URI}/documents/"

    # Ensure content_type is set, fallback to 'application/octet-stream' if None
    content_type = file.content_type or "application/octet-stream"
    files = {"src": (file.filename, file.file, content_type)}
    res = http_post(upload_url, files=files, headers=HEADERS)

    if not res or not res.get("data") or not res["data"].get("id"):
        raise HTTPException(
            status_code=500,
            detail=f"Invalid response from document upload API: {res}",
        )
    return res["data"]["id"]


def create_collection(payload: CollectionCreatePayload) -> str:
    """
    Creates a collection on the external platform.

    Args:
        payload (CollectionCreatePayload): The payload for the API call.

    Returns:
        str: The job ID of the collection creation in progress.
    """
    create_collection_url = f"{BASE_URI}/collections/"
    res = http_post(create_collection_url, json=payload.model_dump(), headers=HEADERS)

    if not res or not res.get("data") or not res["data"].get("job_id"):
        raise HTTPException(
            status_code=500,
            detail=f"Invalid response from collection create API: {res}",
        )
    return res["data"]["job_id"]


def poll_collection_job_status(job_id: str) -> dict:
    """
    Polls the collection job status.

    Args:
        job_id (str): ID of the job to poll.

    Returns:
        dict: The JSON response having the job details.
    """
    status_url = f"{BASE_URI}/collections/jobs/{job_id}"
    final_res = http_get(status_url, headers=HEADERS)

    timeout = TIMEOUT
    start_time = time.time()

    while True:
        if final_res.get("data", {}).get("status") not in ["PENDING", "PROCESSING"]:
            break

        time.sleep(POLLING_INTERVAL)
        final_res = http_get(status_url, headers=HEADERS)

        if time.time() - start_time > timeout:
            break

    if not final_res:
        raise HTTPException(
            status_code=500,
            detail=f"Something went wrong while polling collection job status for job ID {job_id}. Couldn't fetch the response",
        )

    elif final_res.get("data", {}).get("status") in ["PENDING", "PROCESSING"]:
        raise HTTPException(
            status_code=500,
            detail=f"Collection job polling timed out after {timeout} seconds",
        )
    
    elif final_res.get("data", {}).get("status") == "FAILED":
        raise HTTPException(
            status_code=500,
            detail=f"Collection job failed: {final_res.get('error_message')}",
        )
    
    elif final_res.get("success") is False:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch collection job status for job ID {job_id}: {final_res.get('error')}",
        )
    
    logger.info(final_res)

    return final_res.get("data", {}).get("collection", {}) 


def create_and_start_thread(payload: CreateAndStartThreadPayload) -> str:
    """
    Starts a thread to hit the external API for answering a query.

    Args:
        payload (CreateAndStartThreadPayload): The payload for the API call.

    Returns:
        thread_id (str): The ID of the thread created on the external platform.
    """
    thread_url = f"{BASE_URI}/threads/"
    res = http_post(thread_url, json=payload.model_dump(), headers=HEADERS)

    if not res or not res.get("data") or not res["data"].get("thread_id"):
        raise HTTPException(
            status_code=500,
            detail=f"Invalid response from threads API: {res}",
        )
    return res["data"]["thread_id"]


def poll_thread_result(thread_id: str, interval: int = 30, timeout: int = 120) -> str:
    """
    Polls the thread result status.

    Args:
        thread_id (str): ID of the thread to poll.
        interval (int): Polling interval in seconds.
        timeout (int): Maximum time to poll in seconds.

    Returns:
        str: The result/answer from the thread, or raises HTTPException on timeout.
    """
    status_url = f"{BASE_URI}/threads/result/{thread_id}"
    start_time = time.time()

    interval = POLLING_INTERVAL
    timeout = TIMEOUT

    poll_res = None
    final_res = None
    while True:
        time.sleep(interval)
        poll_res = http_get(status_url, headers=HEADERS)

        # Adjust the condition below based on your API's response structure
        if poll_res.get("data", {}).get("status") != "processing":
            final_res = poll_res
            break
        if time.time() - start_time > timeout:
            break

    if not final_res:
        raise HTTPException(
            status_code=500,
            detail=f"Thread result polling timed out after {timeout} seconds. Last response: {poll_res.get('error')}",
        )

    return final_res.get("data", {}).get("response")


def delete_document(document_id: str) -> bool:
    """
    Deletes a document from the external platform.

    Args:
        document_id (str): ID of the document to delete.
    """
    delete_url = f"{BASE_URI}/documents/{document_id}"
    res = http_delete(delete_url, headers=HEADERS)

    if not res.get("success", False):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document {document_id}: {res.get('error')}",
        )

    return True
