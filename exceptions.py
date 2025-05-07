class APIClientError(Exception):
    """Base exception for API client errors."""

    pass


class APIConnectionError(APIClientError):
    """Raised when there's an issue connecting to the API."""

    pass


class APIResponseError(APIClientError):
    """Raised when the API returns an unexpected or error response."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"API Error {status_code}: {message}")


class InvalidAPIKeyError(APIClientError):
    """Raised when an API key is missing or invalid."""

    pass
