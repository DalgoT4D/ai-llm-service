FROM python:3.12-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# install dependencies
COPY pyproject.toml /app
COPY uv.lock /app
RUN uv sync --frozen --no-cache

COPY /src /app/src
COPY /config /app/config
COPY main.py /app/

EXPOSE 7001