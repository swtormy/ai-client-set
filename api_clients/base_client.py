from abc import ABC, abstractmethod
from typing import List, Dict
from exceptions import InvalidAPIKeyError, APIConnectionError, APIResponseError


class BaseAPIClient(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    def send_request(self, messages: List[Dict[str, str]]) -> str:
        pass
