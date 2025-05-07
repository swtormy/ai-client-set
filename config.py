from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your_openai_api_key")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "your_gemini_api_key")
    OLLAMA_API_URL: str = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    DEFAULT_SYSTEM_PROMPT: str = (
        os.getenv("DEFAULT_SYSTEM_PROMPT", 
                  "ты анализатор который помогает сопоставлять спортивные мероприятия. "
                  "тебе нужно возвращать ответ в таком формате: {respone: 0.23} "
                  "c оценкой совпадения от 0.00 до 1.00"
                  )
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
