from typing import List, Dict, Optional, Literal


class ConversationContext:
    def __init__(self, system_prompt: Optional[str] = None):
        self.messages: List[Dict[str, str]] = []
        if system_prompt:
            self.add_message("system", system_prompt)

    def add_message(self, role: Literal["system", "user", "assistant"], content: str):
        """Добавляет сообщение в контекст разговора.
        
        Args:
            role: Роль сообщения ("system", "user", или "assistant")
            content: Текст сообщения
        """
        self.messages.append({"role": role, "content": content})

    def get_context(self, depth: Optional[int] = None) -> List[Dict[str, str]]:
        if depth is None or depth <= 0:
            return list(self.messages)

        actual_messages = list(self.messages)
        system_message = None
        if actual_messages and actual_messages[0]["role"] == "system":
            system_message = actual_messages.pop(0)

        start_index = max(0, len(actual_messages) - depth)
        limited_messages = actual_messages[start_index:]

        if system_message:
            return [system_message] + limited_messages
        return limited_messages

    def clear_context(self, keep_system_prompt: bool = True):
        if keep_system_prompt and self.messages and self.messages[0]["role"] == "system":
            system_prompt_content = self.messages[0]["content"]
            self.messages = [{"role": "system", "content": system_prompt_content}]
        else:
            self.messages = []
