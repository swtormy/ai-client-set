from typing import List, Dict, Optional
from .base_client import BaseAPIClient
from config import settings
from exceptions import (
    InvalidAPIKeyError,
    APIConnectionError,
    APIResponseError,
    APIClientError,
)
from loguru import logger
import google.generativeai as genai


class GeminiClient(BaseAPIClient):
    def __init__(
        self, api_key: str = settings.GEMINI_API_KEY, model_name: str = "gemini-2.0-flash"
    ):
        if not api_key:
            raise InvalidAPIKeyError(
                "Gemini API key not found. Set it in .env or pass it directly."
            )
        super().__init__(api_key)
        self.model_name = model_name
        try:
            genai.configure(api_key=self.api_key)
            self._check_model_availability()
        except Exception as e:
            logger.error(f"GeminiClient: ошибка конфигурации genai или проверки модели: {e}")
            if not isinstance(e, (InvalidAPIKeyError, APIClientError)):
                raise APIClientError(f"Failed to configure Gemini API or check model availability: {e}")
            raise
            
    def _check_model_availability(self):
        try:
            logger.info(f"GeminiClient: Проверка доступности для модели {self.model_name}")
            models_list = self.list_models()
            
            found_model_details = None
            for model_info in models_list:
                if self.model_name == model_info.name or self.model_name == model_info.name.split('/')[-1]:
                    if 'generateContent' in model_info.supported_generation_methods:
                        found_model_details = model_info
                        break
                    else:
                        logger.warning(f"GeminiClient: Модель {model_info.name} найдена, но не поддерживает 'generateContent'.")
            
            if found_model_details:
                logger.info(f"GeminiClient: Модель {found_model_details.name} доступна и поддерживает 'generateContent'.")
                self.model_name = found_model_details.name
            else:
                logger.error(f"GeminiClient: Модель {self.model_name} не найдена среди доступных или не поддерживает 'generateContent'.")
                available_content_models = [m.name for m in models_list if 'generateContent' in m.supported_generation_methods]
                logger.info(f"GeminiClient: Доступные модели (с поддержкой generateContent): {available_content_models}")
                raise APIClientError(f"Model {self.model_name} not found or does not support 'generateContent'.")
        except APIClientError:
            raise
        except Exception as e:
            logger.error(f"GeminiClient: Ошибка при проверке доступности модели {self.model_name}: {e}")
            raise APIClientError(f"Failed to verify model {self.model_name} availability: {e}")

    def list_models(self) -> List[genai.types.Model]:
        try:
            models_iterable = genai.list_models()
            models_list = list(models_iterable)
            logger.info(f"GeminiClient: найдено {len(models_list)} моделей")
            for model_info in models_list:
                logger.debug(f"Доступная модель: {model_info.name}, поддерживает: {model_info.supported_generation_methods}")
            return models_list
        except Exception as e:
            logger.error(f"GeminiClient: ошибка при получении списка моделей: {e}")
            raise APIClientError(f"Failed to list Gemini models: {e}")

    def send_request(self, messages: List[Dict[str, str]]) -> str:
        logger.info(
            f"GeminiClient: отправка запроса к {self.model_name} с {len(messages)} сообщениями."
        )
        logger.debug(f"GeminiClient: сообщения: {messages}")

        try:
            model_obj = genai.GenerativeModel(model_name=self.model_name)
            
            contents_for_api: List[Dict[str, List[str]]] = []
            system_instruction_content: Optional[str] = None

            processed_messages = list(messages)
            if processed_messages and processed_messages[0]["role"] == "system":
                system_instruction_content = processed_messages.pop(0)["content"]

            temp_model_kwargs = {}
            if system_instruction_content:
                if processed_messages and processed_messages[0]["role"] == "user":
                     processed_messages[0]["content"] = f"{system_instruction_content}\n\n{processed_messages[0]["content"]}"
                elif not processed_messages:
                    processed_messages.append({"role": "user", "content": system_instruction_content})
                else:
                    processed_messages.insert(0, {"role": "user", "content": system_instruction_content})

            for msg in processed_messages:
                role = "user" if msg["role"] == "user" else "model"
                contents_for_api.append({"role": role, "parts": [msg["content"]]})

            if not contents_for_api:
                logger.error("GeminiClient: Контекст для API пуст.")
                contents_for_api.append({"role": "user", "parts": ["Hello"]})
            elif contents_for_api[-1]["role"] != "user":
                 logger.warning("GeminiClient: Последнее сообщение в API-контексте не от пользователя. Добавляю 'Продолжай'.")
                 contents_for_api.append({"role": "user", "parts": ["Продолжай"]})

            model_to_use = model_obj

            response = model_to_use.generate_content(
                contents=contents_for_api,
                generation_config=genai.types.GenerationConfig(candidate_count=1)
            )
            
            if response.candidates and response.candidates[0].content.parts:
                result = "".join(part.text for part in response.candidates[0].content.parts)
                logger.info("GeminiClient: успешный ответ получен.")
                return result
            else:
                logger.error(
                    f"GeminiClient: непредвиденная структура ответа или пустой ответ: {response}"
                )
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    error_message = f"Запрос заблокирован: {response.prompt_feedback.block_reason}. {response.prompt_feedback.block_reason_message if response.prompt_feedback.block_reason_message else ''}"
                    logger.error(f"GeminiClient: {error_message}")
                    raise APIResponseError(status_code=0, message=error_message)
                raise APIResponseError(
                    status_code=0,
                    message=f"Непредвиденная структура ответа или пустой ответ от Gemini: {response}",
                )

        except Exception as e:
            logger.exception(f"GeminiClient: непредвиденная ошибка: {e}")
 
            if (
                "API_KEY_INVALID" in str(e)
                or "API_KEY_EXPIRED" in str(e)
                or "API_KEY_BLOCKED" in str(e)
                or "PERMISSION_DENIED" in str(e)
            ):
                raise InvalidAPIKeyError(f"Gemini API key/permission error: {e}")
            
            if "is not found for API version" in str(e) or "Call ListModels" in str(e) or "could not be found" in str(e):
                logger.error(f"GeminiClient: Модель {self.model_name} не найдена или не поддерживает generateContent. {e}")
                raise APIResponseError(status_code=404, message=f"Model {self.model_name} not found or not supported: {e}")

            if isinstance(e, (APIClientError, InvalidAPIKeyError, APIResponseError, APIConnectionError)):
                raise
 
            raise APIClientError(f"Unexpected error in Gemini client: {e}")
