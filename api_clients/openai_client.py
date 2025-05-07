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
import openai


class OpenAIClient(BaseAPIClient):
    def __init__(
        self, api_key: str = settings.OPENAI_API_KEY, model: str = "gpt-3.5-turbo"
    ):
        if not api_key:
            raise InvalidAPIKeyError(
                "OpenAI API key not found. Set it in .env or pass it directly."
            )
        super().__init__(api_key)
        self.model = model
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
        except Exception as e:
            logger.error(f"OpenAIClient: ошибка инициализации клиента OpenAI: {e}")
            raise APIClientError(f"Failed to initialize OpenAI client: {e}")

    def send_request(self, messages: List[Dict[str, str]]) -> str:
        logger.info(
            f"OpenAIClient: отправка запроса к {self.model} с {len(messages)} сообщениями."
        )
        logger.debug(f"OpenAIClient: сообщения: {messages}")

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.content
            ):
                result = response.choices[0].message.content
                logger.info("OpenAIClient: успешный ответ получен.")
                return result
            else:
                logger.error(
                    f"OpenAIClient: непредвиденная структура ответа: {response}"
                )
                raise APIResponseError(
                    status_code=0,
                    message=f"Непредвиденная структура ответа от OpenAI: {response}",
                )

        except openai.APIConnectionError as e:
            logger.error(f"OpenAIClient: ошибка соединения: {e}")
            raise APIConnectionError(f"OpenAI API connection error: {e}")
        except openai.APIStatusError as e:
            logger.error(
                f"OpenAIClient: ошибка статуса API ({e.status_code}): {e.response.text if e.response else str(e)}"
            )
            raise APIResponseError(
                status_code=e.status_code,
                message=e.response.text if e.response else str(e.body or e.message),
            )
        except openai.RateLimitError as e:
            logger.error(f"OpenAIClient: превышен лимит запросов: {e}")
            raise APIResponseError(
                status_code=e.status_code or 429, message=str(e.body or e.message)
            )
        except openai.AuthenticationError as e:
            logger.error(f"OpenAIClient: ошибка аутентификации: {e}")
            raise InvalidAPIKeyError(f"OpenAI API authentication error: {e}")
        except Exception as e:
            logger.exception("OpenAIClient: непредвиденная ошибка.")
            if isinstance(
                e,
                (
                    APIClientError,
                    InvalidAPIKeyError,
                    APIConnectionError,
                    APIResponseError,
                ),
            ):
                raise
            raise APIClientError(f"Unexpected error in OpenAI client: {e}")
