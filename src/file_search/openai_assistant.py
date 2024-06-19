import sys
import math
import time
import json
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
import logging

from openai import OpenAI
from openai.types.beta.assistant import Assistant
from openai.types.beta.threads.message import Message
from openai.types.beta.threads.annotation import Annotation
import pandas as pd


logger = logging.getLogger()


# class OpenAIFileAssistant:
#     _tools = [
#         {
#             "type": "file_search",
#         }
#     ]

#     def __init__(self, openai_key: str, assistant_prompt: str, model: str = "gpt-4o"):
#         self.model = model
#         self.client = OpenAI(api_key=openai_key)
#         self.assistant_prompt = assistant_prompt

#     def query(self, question_prompt: str):
#         logger.info(f"inside the query method with question: {question_prompt}")

#         with Path("app.txt").open("rb") as fp:
#             self.document = self.client.files.create(
#                 file=fp,
#                 purpose="assistants",
#             )
#         self.assistant = self.client.beta.assistants.create(
#             model=self.model,
#             temperature=1e-6,
#             tools=self._tools,
#             instructions=self.assistant_prompt,
#         )
#         self.thread = self.client.beta.threads.create()

#         message = self.client.beta.threads.messages.create(
#             self.thread.id,
#             role="user",
#             content=question_prompt,
#             attachments=[
#                 {
#                     "tools": self._tools,
#                     "file_id": self.document.id,
#                 }
#             ],
#         )

#         run = self.client.beta.threads.runs.create_and_poll(
#             thread_id=self.thread.id,
#             assistant_id=self.assistant.id,
#         )
#         logger.info(f"Status of runnning the thread via create_and_poll : {run.status}")
#         if run.status == "completed":
#             messages: list[Message] = self.client.beta.threads.messages.list(
#                 thread_id=self.thread.id,
#                 run_id=run.id,
#             )
#             logger.info("Query completed")
#             logger.info(messages)
#             body = []
#             for mes in messages:
#                 for content in mes.content:
#                     body.append(content.text.value)
#                     # for a in content.text.annotations:
#                     #     try:
#                     #         document = self[a]
#                     #     except LookupError:
#                     #         continue
#                     #     refn = citations.setdefault(document, len(citations) + 1)
#                     #     body = body.replace(a.text, f" [{refn}]")

#             self.client.beta.threads.messages.delete(
#                 message_id=message.id,
#                 thread_id=self.thread.id,
#             )
#             self.client.files.delete(self.document.id)
#             self.client.beta.threads.delete(self.thread.id)
#             self.client.beta.assistants.delete(self.assistant.id)
#             return body
#         # logger.error("%s (%d): %s", run.status, i + 1, run.last_error)

#         # rest = math.ceil(self.parse_wait_time(run.last_error))
#         # logger.warning("Sleeping %ds", rest)
#         # time.sleep(rest)


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

    def __init__(self, openai_key, log_file, instructions, retries=2, model="gpt-4o"):
        self.retries = retries

        self.client = OpenAI(api_key=openai_key)
        self.parser = AssistantMessage(self.client)

        with Path(log_file).open("rb") as fp:
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.files.delete(self.document.id)
        self.client.beta.threads.delete(self.thread.id)
        self.client.beta.assistants.delete(self.assistant.id)

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
