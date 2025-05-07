import requests
from typing import List, Dict, Optional
from .base_client import BaseAPIClient
from config import settings
from exceptions import APIConnectionError, APIResponseError, APIClientError
from loguru import logger


class OllamaClient(BaseAPIClient):
    def __init__(
        self, model_name: str = "qwen2.5:7b", api_url: str = settings.OLLAMA_API_URL
    ):
        super().__init__(api_key=None)
        self.model_name = model_name
        self.api_url = api_url.rstrip("/") + "/api/generate"

    def send_request(self, messages: List[Dict[str, str]]) -> str:
        logger.info(
            f"OllamaClient: отправка запроса к {self.model_name} ({self.api_url}) с {len(messages)} сообщениями."
        )
        logger.debug(f"OllamaClient: сообщения: {messages}")

        system_prompt_content: Optional[str] = None
        processed_messages = list(messages)

        if processed_messages and processed_messages[0]["role"] == "system":
            system_prompt_content = processed_messages.pop(0)["content"]

        prompt_parts = []
        for msg in processed_messages:
            role = msg["role"].capitalize()
            prompt_parts.append(f"{role}: {msg['content']}")

        full_prompt = "\n".join(prompt_parts)

        if not full_prompt and system_prompt_content:

            logger.warning(
                "OllamaClient: Входящие сообщения содержат только системный промпт или пусты. Используется 'Hello'."
            )
            full_prompt = "Hello"

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
        }

        if system_prompt_content:
            payload["system"] = system_prompt_content

        logger.debug(f"OllamaClient: payload: {payload}")

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            logger.error(f"OllamaClient: таймаут при запросе к {self.api_url}: {e}")
            raise APIConnectionError(
                f"Timeout connecting to Ollama at {self.api_url}: {e}"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"OllamaClient: ошибка соединения с {self.api_url}: {e}")
            raise APIConnectionError(
                f"Error connecting to Ollama at {self.api_url}: {e}"
            )

        try:
            data = response.json()
            logger.debug(f"OllamaClient: получен ответ: {data}")

            if "response" in data and data.get("response"):
                logger.info("OllamaClient: успешный ответ получен.")
                return str(data["response"]).strip()
            elif "error" in data:
                error_msg = data["error"]
                logger.error(f"OllamaClient: API вернуло ошибку: {error_msg}")
                raise APIResponseError(
                    status_code=response.status_code,
                    message=f"Ollama API error: {error_msg}",
                )
            else:
                logger.error(f"OllamaClient: непредвиденная структура ответа: {data}")
                raise APIResponseError(
                    status_code=response.status_code,
                    message=f"Unexpected response structure from Ollama: {data}",
                )
        except ValueError as e:
            logger.error(
                f"OllamaClient: ошибка декодирования JSON ответа: {e}. Ответ: {response.text}"
            )
            raise APIResponseError(
                status_code=response.status_code,
                message=f"Failed to decode JSON response from Ollama: {e}. Response text: {response.text[:200]}",
            )
        except Exception as e:
            logger.exception(
                f"OllamaClient: непредвиденная ошибка при обработке ответа: {e}"
            )
            raise APIClientError(f"Unexpected error processing Ollama response: {e}")
