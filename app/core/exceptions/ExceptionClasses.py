from typing import Any

from .ErrorCode import ErrorCode


class NgAutomateException(Exception):
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class ValidationError(NgAutomateException):
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.INVALID_INPUT, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, error_code, status_code=400, details=details)


class AuthenticationError(NgAutomateException):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, ErrorCode.INVALID_CREDENTIALS, status_code=401, details=details)


class AuthorizationError(NgAutomateException):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, ErrorCode.INSUFFICIENT_PERMISSIONS, status_code=403, details=details)


class NotFoundError(NgAutomateException):
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.RESOURCE_NOT_FOUND, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, error_code, status_code=404, details=details)


class PayloadTooLargeError(NgAutomateException):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, ErrorCode.PAYLOAD_TOO_LARGE, status_code=413, details=details)


class GitFetchError(NgAutomateException):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, ErrorCode.GIT_FETCH_FAILED, status_code=422, details=details)


class SandboxError(NgAutomateException):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, ErrorCode.SANDBOX_FAILED, status_code=503, details=details)


class AnalysisError(NgAutomateException):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, ErrorCode.ANALYSIS_FAILED, status_code=422, details=details)
