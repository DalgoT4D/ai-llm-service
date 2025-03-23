# ai-llm-service
A lighweight service to serve ai/llm needs. All requests are queued as tasks and executed with some retry strategy by celery worker(s)

# UV package manager

The project uses `uv` as its package manager. You will need to install it on your machine

UV can be installed system-wide using cURL on macOS and Linux:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sudo sh
```

And with Powershell on Windows (make sure you run Powershell with administrator privileges):

```sh
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

UV is available via Homebrew as well:

```sh
brew install uv
```

# Docker setup


1. Clone the repository:
```sh
git clone https://github.com/DalgoT4D/ai-llm-service.git
```

2. Navigate to the project directory:
```sh
cd ai-llm-service
```

3. Start the services
```sh
docker compose up --build
```

# Bare setup

To run the `ai-llm-service` project, follow these steps:

1. Clone the repository:
```sh
git clone https://github.com/DalgoT4D/ai-llm-service.git
```

2. Navigate to the project directory:
```sh
cd ai-llm-service
```

3. Install the required dependencies:
```sh
uv sync
```

5. Setup your .env file, Make sure you have a redis server running
```sh
cp .env.example .env
```

Update the relevant fields in `.env`

5. Start the Celery worker(s):
```sh
uv run celery -A main.celery worker -n llm -Q llm --loglevel=INFO
```

6. Monitor your celery tasks and queues using flower:
```sh
uv run celery -A main.celery flower --port=5555
```
Dashboard will be available at `http://localhost:5555`

7. Start the FastAPI server:
```sh
uv run main.py
```

# UV package management

1. To add new package using uv

```sh
uv add <package_name>
```

2. To remove a package using uv

```sh
uv remove <package_name>
```

# Features supported

## File search
Currently the service supports the openai's file search but can be easily extended to other services. The request response flow here is as follows
1. Client uploads a file (to query on) to the service.

2. Client uses the `file_path` from 1. to query. Note the client needs to provided with a `system_prompt` or an `assistant_prompt`. Client can do multiple queries here

3. Client polls for the response until the job/task reaches a terminal state.

4. Client gets the result with a `session_id`. Client can either continue querying the same file or close the session

## API

API documentation can be found at https://llm.projecttech4dev.org/docs

Local docs can be found at http://127.0.0.1:7001/docs
