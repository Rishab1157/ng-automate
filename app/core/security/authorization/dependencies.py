from collections.abc import Callable

from bson import ObjectId
from fastapi import Depends, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.exceptions import AuthorizationError, ErrorMessages, ValidationError
from app.core.security.authentication import decode_access_token
from app.models.authModel import CurrentUserMapper, CurrentUserModel
from app.permissions.ngAutomatePermission import NGAUTOMATE_ACCESS_ALL_ORGS

security = HTTPBearer(description="QXcel access token")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> CurrentUserModel:
    return CurrentUserMapper.from_token_payload(decode_access_token(credentials.credentials))


def require_permission(required: list[str]) -> Callable[..., CurrentUserModel]:
    """Dependency: the caller must hold at least one of the permission codes."""

    async def dependency(current_user: CurrentUserModel = Depends(get_current_user)) -> CurrentUserModel:
        if not current_user.has_any_permission(required):
            raise AuthorizationError(ErrorMessages.MISSING_PERMISSION)
        return current_user

    return dependency


async def resolve_org_id(
    target_org_id: str | None = Query(None, description="Organization ID (defaults to caller's org)"),
    current_user: CurrentUserModel = Depends(get_current_user),
) -> str:
    """The organization a request works in.

    Unlike QXcel's resolve_org_scope, a target org the caller may not act for is refused
    (403) instead of silently replaced by the caller's own org.
    """
    if not target_org_id or target_org_id == current_user.org_id:
        return current_user.org_id
    if not ObjectId.is_valid(target_org_id):
        raise ValidationError(ErrorMessages.INVALID_OBJECT_ID.format(field="target_org_id"))
    if not current_user.has_permission(NGAUTOMATE_ACCESS_ALL_ORGS):
        raise AuthorizationError(ErrorMessages.CROSS_ORG_FORBIDDEN)
    return target_org_id
