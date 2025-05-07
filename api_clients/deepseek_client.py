from typing import List, Dict
from .base_client import BaseAPIClient
from config import settings
from exceptions import (
    InvalidAPIKeyError,
    APIConnectionError,
    APIResponseError,
    APIClientError,
)
from loguru import logger
import httpx

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekClient(BaseAPIClient):
    def __init__(
        self, api_key: str = settings.DEEPSEEK_API_KEY, model: str = "deepseek-chat"
    ):
        if not api_key:
            raise InvalidAPIKeyError(
                "DeepSeek API key not found. Set it in .env or pass it directly."
            )
        super().__init__(api_key)
        self.model = model

    def send_request(self, messages: List[Dict[str, str]]) -> str:
        logger.info(
            f"DeepSeekClient: отправка запроса к {self.model} с {len(messages)} сообщениями."
        )
        logger.debug(f"DeepSeekClient: сообщения: {messages}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {"model": self.model, "messages": messages, "stream": False}

        try:
            with httpx.Client() as client:
                response = client.post(
                    DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30.0
                )
                response.raise_for_status()

            response_data = response.json()
            if response_data.get("choices") and len(response_data["choices"]) > 0:
                message = response_data["choices"][0].get("message")
                if message and message.get("content"):
                    result = message["content"]
                    logger.info("DeepSeekClient: успешный ответ получен.")
                    return result

            logger.error(
                f"DeepSeekClient: непредвиденная структура ответа: {response_data}"
            )
            raise APIResponseError(
                status_code=response.status_code,
                message=f"Непредвиденная структура ответа: {response_data}",
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                f"DeepSeekClient: ошибка HTTP статуса {e.response.status_code}: {e.response.text}"
            )
            raise APIResponseError(
                status_code=e.response.status_code, message=e.response.text
            )
        except httpx.RequestError as e:
            logger.error(f"DeepSeekClient: ошибка соединения или запроса: {e}")
            raise APIConnectionError(f"DeepSeek API connection/request error: {e}")
        except Exception as e:
            logger.exception("DeepSeekClient: непредвиденная ошибка.")
            raise APIClientError(f"Unexpected error in DeepSeek client: {e}")
