import requests
import logging
from fastapi import HTTPException

logger = logging.getLogger()


def http_post(endpoint: str, json: dict = None, files: dict = None, **kwargs) -> dict:
    """make a POST request"""
    headers = kwargs.pop("headers", {})
    timeout = kwargs.pop("timeout", None)

    try:
        res = requests.post(
            endpoint,
            headers=headers,
            timeout=timeout,
            json=json,
            files=files,
            **kwargs,
        )
    except Exception as error:
        logger.exception(error)
        raise HTTPException(500, "connection error") from error
    try:
        res.raise_for_status()
    except Exception as error:
        logger.exception(error)
        raise HTTPException(res.status_code, res.text) from error
    return res.json()


def http_get(endpoint: str, **kwargs) -> dict:
    """make a GET request"""
    headers = kwargs.pop("headers", {})
    timeout = kwargs.pop("timeout", None)

    try:
        res = requests.get(endpoint, headers=headers, timeout=timeout, **kwargs)
    except Exception as error:
        logger.exception(error)
        raise HTTPException(500, "connection error") from error
    try:
        res.raise_for_status()
    except Exception as error:
        logger.exception(error)
        raise HTTPException(res.status_code, res.text) from error
    return res.json()


def http_delete(endpoint: str, **kwargs) -> dict:
    """make a DELETE request"""
    headers = kwargs.pop("headers", {})
    timeout = kwargs.pop("timeout", None)

    try:
        res = requests.delete(endpoint, headers=headers, timeout=timeout, **kwargs)
    except Exception as error:
        logger.exception(error)
        raise HTTPException(500, "connection error") from error
    try:
        res.raise_for_status()
    except Exception as error:
        logger.exception(error)
        raise HTTPException(res.status_code, res.text) from error
    return res.json()
