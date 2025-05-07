from fastapi import FastAPI, HTTPException, Body
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
import uvicorn

from api_clients.base_client import BaseAPIClient
from api_clients.openai_client import OpenAIClient
from api_clients.deepseek_client import DeepSeekClient
from api_clients.gemini_client import GeminiClient
from api_clients.ollama_client import OllamaClient
from context_manager import ConversationContext
from config import settings
from exceptions import (
    APIClientError,
    InvalidAPIKeyError,
    APIResponseError,
    APIConnectionError,
)
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("api_server.log", rotation="10 MB", retention="7 days", level="DEBUG")

app = FastAPI(
    title="AI Chat API",
    description="API для взаимодействия с различными моделями ИИ.",
    version="1.0.0",
)


class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    client_type: Literal["openai", "deepseek", "gemini", "ollama"] = Field(
        ..., description="Тип клиента для использования."
    )
    messages: List[ChatMessageInput] = Field(
        ..., description="Список сообщений в диалоге."
    )
    model_name_override: Optional[str] = Field(
        None, description="Имя модели для переопределения (для Gemini, Ollama)."
    )
    system_prompt_override: Optional[str] = Field(
        None,
        description="Системное сообщение для переопределения настроек по умолчанию.",
    )
    context_depth: Optional[int] = Field(
        None, description="Глубина контекста. 0 или None для полного контекста."
    )


class ChatResponse(BaseModel):
    assistant_response: str
    client_used: str
    model_used: Optional[str] = None


class ErrorDetail(BaseModel):
    detail: str


def get_api_client(
    client_type: str, model_name_override: Optional[str] = None
) -> BaseAPIClient:
    logger.info(
        f"API: Попытка создать клиента: {client_type}, модель: {model_name_override}"
    )
    client_type = client_type.strip().lower()

    try:
        if client_type == "openai":
            return OpenAIClient()
        elif client_type == "deepseek":
            return DeepSeekClient()
        elif client_type == "gemini":
            return (
                GeminiClient(model_name=model_name_override)
                if model_name_override
                else GeminiClient()
            )
        elif client_type == "ollama":
            return (
                OllamaClient(model_name=model_name_override)
                if model_name_override
                else OllamaClient()
            )
        else:
            raise ValueError(f"Неизвестный тип клиента: {client_type}")
    except InvalidAPIKeyError as e:
        logger.error(f"API: Ошибка API ключа для клиента {client_type}: {e}")
        raise
    except APIClientError as e:
        logger.error(f"API: Ошибка конфигурации клиента {client_type}: {e}")
        raise
    except Exception as e:
        logger.error(
            f"API: Непредвиденная ошибка при создании клиента {client_type}: {e}"
        )
        raise APIClientError(f"Не удалось создать клиент {client_type}: {e}")


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorDetail, "description": "Некорректный запрос"},
        401: {"model": ErrorDetail, "description": "Ошибка API ключа"},
        404: {
            "model": ErrorDetail,
            "description": "Модель не найдена или не поддерживается",
        },
        500: {"model": ErrorDetail, "description": "Внутренняя ошибка сервера"},
        503: {
            "model": ErrorDetail,
            "description": "Ошибка соединения с API провайдера",
        },
    },
)
async def handle_chat_request(request: ChatRequest = Body(...)):
    logger.info(f"API: Получен запрос для клиента: {request.client_type}")
    logger.debug(f"API: Входящий запрос: {request.model_dump_json(indent=2)}")

    try:
        api_client = get_api_client(request.client_type, request.model_name_override)
    except InvalidAPIKeyError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except APIClientError as e:
        if "not found" in str(e).lower() or "404" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(
            status_code=400, detail=f"Ошибка конфигурации клиента: {str(e)}"
        )
    except Exception as e:
        logger.exception("API: Непредвиденная ошибка при инициализации клиента.")
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка при инициализации клиента: {str(e)}",
        )

    system_prompt_to_use: Optional[str] = None
    input_messages: List[Dict[str, str]] = []

    if request.system_prompt_override is not None:
        system_prompt_to_use = request.system_prompt_override
        for msg in request.messages:
            if msg.role != "system":
                input_messages.append({"role": msg.role, "content": msg.content})
    else:

        found_system_in_messages = False
        for msg in request.messages:
            if msg.role == "system":
                if not found_system_in_messages:
                    system_prompt_to_use = msg.content
                    found_system_in_messages = True
                else:
                    logger.warning(
                        "API: Найдено несколько системных сообщений во входящем списке, используется первое."
                    )
            else:
                input_messages.append({"role": msg.role, "content": msg.content})

        if not found_system_in_messages:
            system_prompt_to_use = settings.DEFAULT_SYSTEM_PROMPT

    if (
        not input_messages
        and system_prompt_to_use is not None
        and not system_prompt_to_use.strip()
    ):

        pass

    context = ConversationContext()
    if system_prompt_to_use and system_prompt_to_use.strip():
        context.add_message("system", system_prompt_to_use)
        logger.info(
            f"API: Используется системное сообщение: '{system_prompt_to_use[:100]}...'"
        )
    else:
        logger.info("API: Системное сообщение не используется или пустое.")

    for msg_data in input_messages:
        context.add_message(msg_data["role"], msg_data["content"])

    current_context_for_api = context.get_context(depth=request.context_depth)

    if not current_context_for_api or all(
        m["role"] == "system" for m in current_context_for_api
    ):

        logger.warning(
            "API: Контекст для API пуст или содержит только системное сообщение после обработки."
        )

        if not any(m["role"] == "user" for m in current_context_for_api):
            raise HTTPException(
                status_code=400,
                detail="Запрос должен содержать хотя бы одно сообщение от пользователя.",
            )

    try:
        logger.info(
            f"API: Отправка запроса к {api_client.model_name if hasattr(api_client, 'model_name') else request.client_type}"
        )
        response_text = await run_in_threadpool(
            api_client.send_request, current_context_for_api
        )

        logger.info(f"API: Ответ от ассистента получен.")
        return ChatResponse(
            assistant_response=response_text,
            client_used=request.client_type,
            model_used=getattr(api_client, "model_name", None),
        )
    except APIResponseError as e:
        logger.error(f"API: Ошибка ответа от API провайдера: {e}")
        status_code = 500
        if hasattr(e, "status_code") and isinstance(e.status_code, int):
            if e.status_code == 401 or e.status_code == 403:
                status_code = 401
            elif e.status_code == 404:
                status_code = 404
            elif e.status_code == 429:
                status_code = 429
            elif 400 <= e.status_code < 500:
                status_code = 400
        raise HTTPException(status_code=status_code, detail=str(e))
    except APIConnectionError as e:
        logger.error(f"API: Ошибка соединения с API провайдера: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("API: Непредвиденная внутренняя ошибка при обработке запроса.")
        raise HTTPException(
            status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


async def main_async():
    config = uvicorn.Config("api_server:app", host="0.0.0.0", port=8000, reload=True)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
