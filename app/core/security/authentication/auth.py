from typing import Any

import jwt

from app.config import settings
from app.core.exceptions import AuthenticationError, ErrorMessages


def decode_access_token(token: str) -> dict[str, Any]:
    """Verify a QXcel access token (signature and expiry) and return its payload."""
    try:
        return jwt.decode(
            token,
            settings.SECRET_KEY.get_secret_value(),
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["exp"]},
        )
    except jwt.PyJWTError:
        # One message for every failure: the caller learns nothing about why.
        raise AuthenticationError(ErrorMessages.INVALID_TOKEN) from None
