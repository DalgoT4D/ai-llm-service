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
from openai.types.beta.threads.message import Message
from openai.types.beta.threads.annotation import Annotation
import pandas as pd

from src.file_search.session import OpenAISessionState, FileSearchSession


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
        file_path: str = None,
        instructions: str = None,
        session_id: str = None,
        retries=2,
        model="gpt-4o",
    ):
        curr_session = None
        if session_id:
            curr_session: OpenAISessionState = FileSearchSession.get(session_id)
        self.retries = retries
        self.client = OpenAI(api_key=openai_key)
        self.parser = AssistantMessage(self.client)

        if curr_session:
            logger.info(f"Resuming session {curr_session.id}")
            self.document = self.client.files.retrieve(curr_session.document_id)
            self.assistant = self.client.beta.assistants.retrieve(
                curr_session.assistant_id
            )
            self.thread = self.client.beta.threads.retrieve(curr_session.thread_id)
        else:
            logger.info("Creating a new session")
            with Path(file_path).open("rb") as fp:
                self.document = self.client.files.create(
                    file=fp,
                    purpose="assistants",
                )
            self.assistant = self.client.beta.assistants.create(
                model=model,
                temperature=1e-6,
                tools=self._tools,
                instructions=instructions,
            )
            self.thread = self.client.beta.threads.create()
            # create a new session
            curr_session = OpenAISessionState(
                id=str(uuid.uuid4()),
                document_id=self.document.id,
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
                local_fpath=file_path,
            )
            FileSearchSession.set(curr_session.id, curr_session)

        self.session = curr_session

    def query(self, content):
        message = self.client.beta.threads.messages.create(
            self.thread.id,
            role="user",
            content=content,
            attachments=[
                {
                    "tools": self._tools,
                    "file_id": self.document.id,
                }
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
        self.client.files.delete(self.document.id)
        self.client.beta.threads.delete(self.thread.id)
        self.client.beta.assistants.delete(self.assistant.id)
        FileSearchSession.remove(self.session.id)
        if self.session.local_fpath:
            Path(self.session.local_fpath).unlink()
