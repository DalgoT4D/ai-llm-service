# ai-llm-service
A lighweight service to serve ai/llm needs. All requests are queued as tasks and executed with some retry strategy by celery worker(s)

# Setup

To run the `ai-llm-service` project, follow these steps:

1. Clone the repository:
    ```
    git clone https://github.com/DalgoT4D/ai-llm-service.git
    ```

2. Navigate to the project directory:
    ```
    cd ai-llm-service
    ```

3. Create a virtual environment and activate it:
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```

4. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

5. Setup your .env file, Make sure you have a redis server running
    ```
    cp .env.example .env
    ```

    Update the relevant fields in `.env`

5. Start the Celery worker(s):
    ```
    celery -A main.celery worker -n llm -Q llm --loglevel=INFO
    ```

6. Monitor your celery tasks and queues using flower:
    ```
    celery -A main.celery flower --port=5555
    ```
    Dashboard will be available at `http://localhost:5555`

7. Start the FastAPI server:
    
    Dev server
    ```
    python3 main.py
    ```

You can test the service by sending requests to the available endpoints.

# Features supported

## File search
Currently the service supports the openai's file search but can be easily extended to other services. The request response flow here is as follows
1. Client uploads a file (to query on) to the service.

2. Client uses the `file_path` from 1. to query. Note the client needs to provided with a `system_prompt` or an `assistant_prompt`. Client can do multiple queries here

3. Client polls for the response until the job/task reaches a terminal state.

4. Client gets the result with a `session_id`. Client can either continue querying the same file or close the session

## API

All APIs are async and return a `task_id`.

####  `GET` `/task/{task_id}`

Used to fetch status of an async task

Returns
- `id`
- `status`
- `result`
- `error`


#### `POST` `/file/upload`

Uploads a file to be queried against

Request format
- `filename`
- `file` (bytes)


####  `POST` `/file/query`

Queries an uploaded file

Request format
- `file_path`
- `assistant_prompt`
- `queries` (list)
- `session_id`

Returns
- `task_id`


####  `DELETE` `/file/search/session/{session_id}`

Closes a session

Returns
- `task_id`

