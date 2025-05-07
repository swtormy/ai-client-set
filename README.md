
*   **Мульти-клиентская поддержка через API:**
    *   OpenAI (модели GPT)
    *   DeepSeek (модели DeepSeek)
    *   Google Gemini (с возможностью указать конкретную модель, например, `gemini-1.5-pro`, `gemini-2.0-flash`)
    *   Ollama (локальные модели, например, `qwen2.5:7b`; имя модели можно указать в запросе).
*   **Проверка модели Gemini:** Автоматическая проверка доступности указанной модели Gemini и её поддержки метода `generateContent` при инициализации клиента.
*   **Управление контекстом:** 
    *   Поддержка контекста беседы с возможностью настройки глубины истории через API запрос.
    *   **Системное сообщение:** Может быть задано через запрос API (`system_prompt_override`), либо взято из тела запроса (`messages` с `role: "system"`), либо использовано значение по умолчанию из `DEFAULT_SYSTEM_PROMPT` в `.env`.
*   **Конфигурация:** 
    *   API ключи (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`), `DEFAULT_SYSTEM_PROMPT` и `OLLAMA_API_URL` настраиваются через переменные окружения в файле `.env`.
*   **Логирование:**
    *   Запись логов уровня `DEBUG` и выше в файл `api_server.log`.
    *   Вывод логов уровня `INFO` и выше в консоль при работе сервера.
*   **Настройки безопасности Gemini:** Для клиента Google Gemini стандартные фильтры безопасности (`HarmCategory`) отключены.


```
mapper-assistent/
├── api_clients/                # Клиенты для взаимодействия с API
│   ├── __init__.py
│   ├── base_client.py          # Абстрактный базовый класс
│   ├── deepseek_client.py      # Клиент DeepSeek
│   ├── gemini_client.py        # Клиент Google Gemini
│   ├── ollama_client.py        # Клиент Ollama
│   └── openai_client.py        # Клиент OpenAI
├── .env.example                # Пример файла конфигурации (.env)
├── .gitignore                  
├── api_server.log              # Файл логов API сервера (создается автоматически)
├── config.py                   # Загрузка конфигурации
├── context_manager.py          # Управление контекстом
├── exceptions.py               # Пользовательские исключения
├── main.py                     # Основной файл FastAPI сервера 
├── README.md                   # Данный файл
└── requirements.txt            # Зависимости
```



**Настройки переменных окружения:**
    Создайте файл `.env` в корневой директории проекта, скопировав содержимое из `.env.example` и вставив ваши API ключи и другие настройки:
    ```env
    OPENAI_API_KEY="your_openai_api_key"
    DEEPSEEK_API_KEY="your_deepseek_api_key"
    GEMINI_API_KEY="your_gemini_api_key"
    OLLAMA_API_URL="http://localhost:11434" # Если ваш Ollama на другом URL
    DEFAULT_SYSTEM_PROMPT="ты анализатор который помогает сопоставлять спортивные мероприятия..."
    ```


POST запросы на эндпоинт `/api/v1/chat`.

**Тело запроса (JSON):**
```json
{
  "client_type": "ollama", // "openai", "deepseek", "gemini", "ollama"
  "messages": [
    {"role": "system", "content": "Be concise."}, // Опционально, если не используется system_prompt_override
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "model_name_override": "qwen2.5:7b", // Опционально для gemini, ollama
  "system_prompt_override": null, // Опционально, имеет приоритет над system в messages и DEFAULT_SYSTEM_PROMPT
  "context_depth": 10 // Опционально
}
```


**Успешный ответ (JSON):**
```json
{
  "assistant_response": "Paris",
  "client_used": "ollama",
  "model_used": "qwen2.5:7b"
}
```
