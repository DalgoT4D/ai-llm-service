import sys
import math
import time
import json
import uuid
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
import logging
import io

from openai import OpenAI
from openai.types.beta.assistant import Assistant
from openai.types.beta.thread import Thread
from openai.types.file_object import FileObject
from openai.types.beta.threads.message import Message
from openai.types.beta.threads.annotation import Annotation
import pandas as pd

from src.file_search.session import (
    OpenAISessionState,
    FileSearchSession,
    SessionStatusEnum,
)


logger = logging.getLogger()


class AssistantMessage:
    _ctypes = (
        "file_citation",
        "file_path",
    )

    def __init__(self, client):
        self.client = client

    def __getitem__(self, item):
        for c in self._ctypes:
            if hasattr(item, c):
                citation = getattr(item, c)
                reference = self.client.files.retrieve(citation.file_id)
                return reference.filename

        raise LookupError()

    def __call__(self, message):
        citations = {}

        for m in message:
            for c in m.content:
                body = c.text.value

                for a in c.text.annotations:
                    try:
                        document = self[a]
                    except LookupError:
                        continue
                    refn = citations.setdefault(document, len(citations) + 1)
                    body = body.replace(a.text, f" [{refn}]")

                if citations:
                    iterable = (f"[{y}] {x}" for (x, y) in citations.items())
                    citestr = "\n\n{}".format("\n".join(iterable))
                    citations.clear()
                else:
                    citestr = ""

                yield f"{body}{citestr}"

    def to_string(self, message):
        return "\n".join(self(message))


class OpenAIFileAssistant:
    _tools = [
        {
            "type": "file_search",
        }
    ]

    @staticmethod
    def parse_wait_time(err):
        if err.code == "rate_limit_exceeded":
            for i in err.message.split(". "):
                if i.startswith("Please try again in"):
                    (*_, wait) = i.split()
                    return pd.to_timedelta(wait).total_seconds()

        raise TypeError(err.code)

    def __init__(
        self,
        openai_key: str,
        session_id: str,
        instructions: str = None,
        retries=2,
        model="gpt-4o",
    ):
        curr_session: OpenAISessionState = FileSearchSession.get(session_id)
        if not curr_session:
            raise ValueError("Session not found")
        self.retries = retries
        self.client = OpenAI(api_key=openai_key)
        self.parser = AssistantMessage(self.client)

        self.documents: list[FileObject] = []
        if curr_session.status == SessionStatusEnum.locked:
            logger.info(
                f"Resuming session {curr_session.id}; retrieving references to openai & loading them in memory"
            )
            self.documents = [
                self.client.files.retrieve(doc_id)
                for doc_id in curr_session.document_ids
            ]
            self.assistant = self.client.beta.assistants.retrieve(
                curr_session.assistant_id
            )
            self.thread = self.client.beta.threads.retrieve(curr_session.thread_id)
        else:
            logger.info(
                "Uploading documents to openai for the first time; setting the session to locked"
            )
            for file_path in curr_session.local_fpaths:
                with Path(file_path).open("rb") as fp:
                    uploaded_doc = self.client.files.create(
                        file=fp,
                        purpose="assistants",
                    )
                    self.documents.append(uploaded_doc)

            curr_session.document_ids = [
                uploaded_doc.id for uploaded_doc in self.documents
            ]

            self.assistant: Assistant = self.client.beta.assistants.create(
                model=model,
                temperature=1e-6,
                tools=self._tools,
                instructions=instructions,
            )
            curr_session.assistant_id = self.assistant.id

            self.thread: Thread = self.client.beta.threads.create()
            curr_session.thread_id = self.thread.id

            # update in redis
            curr_session.status = SessionStatusEnum.locked
            FileSearchSession.set(curr_session.id, curr_session)

        self.session = curr_session

    def query(self, content):
        message = self.client.beta.threads.messages.create(
            self.thread.id,
            role="user",
            content=content,
            attachments=[
                {"tools": self._tools, "file_id": doc.id} for doc in self.documents
            ],
        )

        for i in range(self.retries):
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
            )
            if run.status == "completed":
                break
            logger.error("%s (%d): %s", run.status, i + 1, run.last_error)

            rest = math.ceil(self.parse_wait_time(run.last_error))
            logger.warning("Sleeping %ds", rest)
            time.sleep(rest)
        else:
            raise TimeoutError("Message retries exceeded")

        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id,
            run_id=run.id,
        )
        self.client.beta.threads.messages.delete(
            message_id=message.id,
            thread_id=self.thread.id,
        )

        return self.parser.to_string(messages)

    def close(self):
        for doc in self.documents:
            self.client.files.delete(doc.id)
        self.client.beta.threads.delete(self.thread.id)
        self.client.beta.assistants.delete(self.assistant.id)
        for local_fpath in self.session.local_fpaths:
            Path(local_fpath).unlink()
        # remove from redis
        FileSearchSession.remove(self.session.id)
