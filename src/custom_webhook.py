import requests
from requests.exceptions import HTTPError
from pydantic import BaseModel
import logging


logger = logging.getLogger()


class WebhookConfig(BaseModel):
    """
    Details of webhook required to post results
    """

    endpoint: str
    headers: dict


class CustomWebhook:
    timeout = 20

    def __init__(self, config: WebhookConfig):
        # TODO: maybe some validations on the endpoint etc.
        self.config: WebhookConfig = config

    def post_result(self, results: dict):
        """
        Posts data to the configured webhook endpoint.
        """
        try:
            response = requests.post(
                self.config.endpoint,
                json=results,
                headers=self.config.headers,
                timeout=self.timeout,
            )

            response.raise_for_status()

            logging.info(f"Successfully posted results to {self.config.endpoint}")
            return response.json()
        except HTTPError as err:
            logging.error(f"Failed to post webhook results {err.response.text}")
            return {"error": str(err.response.text)}
        except Exception as err:
            logging.error(
                f"Failed to post webhook results {str(err)}. Something went wrong"
            )
            return {"error": str(err)}
