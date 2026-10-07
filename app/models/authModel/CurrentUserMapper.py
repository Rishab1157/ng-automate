from typing import Any

from bson import ObjectId

from app.core.exceptions import AuthenticationError, ErrorMessages

from .CurrentUserModel import CurrentUserModel


class CurrentUserMapper:
    @staticmethod
    def from_token_payload(payload: dict[str, Any]) -> CurrentUserModel:
        user_id = str(payload.get("user_id") or "")
        org_id = str(payload.get("org_id") or "")
        if not (ObjectId.is_valid(user_id) and ObjectId.is_valid(org_id)):
            raise AuthenticationError(ErrorMessages.INVALID_TOKEN_PAYLOAD)

        permissions = payload.get("permissions")
        return CurrentUserModel(
            user_id=user_id,
            org_id=org_id,
            role_id=payload.get("role_id"),
            email=payload.get("email") or "",
            permissions=frozenset(permissions) if isinstance(permissions, list) else frozenset(),
        )
