import os
from pathlib import Path
from flask import Flask, request, make_response, jsonify
from src.api import text_summ
from logging.config import dictConfig
import logging


app = Flask(__name__)

log_dir = Path(__file__).resolve().parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# logging configuration
dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S %Z",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/app.log",
                "maxBytes": 1048576,  # 1MB
                "backupCount": 5,
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)


# auth middleware
@app.before_request
def check_auth():
    auth_token = request.headers.get("Authorization")
    if not auth_token or auth_token != os.getenv("API_KEY"):
        logging.info("Request authenticated")
        return make_response({"message": "unauthorized"}, 401)
    logging.info("successfully authenticated")


# register routes
app.register_blueprint(text_summ)


# home route
@app.route("/", methods=["GET"])
def home():
    return jsonify(message="Hello, World!")


if __name__ == "__main__":
    app.run(debug=True, load_dotenv=True, port=7000)
