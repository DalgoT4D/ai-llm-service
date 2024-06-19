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
    celery -A main.celery worker -n llm --loglevel=info -Q llm
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
