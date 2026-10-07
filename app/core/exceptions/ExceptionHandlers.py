"""Turn every error into QXcel's error shape: {"error": {code, message, details, status_code}}."""

import logging
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .ErrorCode import ErrorCode
from .ErrorMessages import ErrorMessages
from .ExceptionClasses import NgAutomateException

logger = logging.getLogger(__name__)

_CODE_BY_STATUS = {
    400: ErrorCode.INVALID_INPUT,
    401: ErrorCode.INVALID_CREDENTIALS,
    403: ErrorCode.INSUFFICIENT_PERMISSIONS,
    404: ErrorCode.RESOURCE_NOT_FOUND,
    413: ErrorCode.PAYLOAD_TOO_LARGE,
}


def _error_response(
    status_code: int,
    code: ErrorCode,
    message: str,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = {"error": {"code": code.value, "message": message, "details": details or {}, "status_code": status_code}}
    return JSONResponse(status_code=status_code, content=body, headers=headers)


async def ng_automate_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, NgAutomateException)
    headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
    return _error_response(exc.status_code, exc.error_code, exc.message, exc.details, headers)


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = _CODE_BY_STATUS.get(exc.status_code, ErrorCode.INTERNAL_SERVER_ERROR)
    return _error_response(exc.status_code, code, str(exc.detail), headers=exc.headers)


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    errors = [
        {"field": ".".join(str(part) for part in error["loc"]), "message": error["msg"]}
        for error in exc.errors()
    ]
    return _error_response(422, ErrorCode.INVALID_INPUT, "Request validation failed", {"errors": errors})


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Full details go to the log only; the caller never sees internals.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _error_response(500, ErrorCode.INTERNAL_SERVER_ERROR, ErrorMessages.INTERNAL_ERROR)
