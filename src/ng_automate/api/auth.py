"""Who is calling, and which organization they may act for.

Tokens come from QXcel: HS256 JWTs signed with the shared SECRET_KEY.
Access is decided only by the exact permission codes inside the token.
"""

import jwt
from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ng_automate.core.env import require_env
from ng_automate.models.auth import CurrentUser

ALGORITHM = "HS256"
ACCESS_ALL_ORGS = "NGAUTOMATE:ACCESS:ALL_ORGS"
ACCESS_OWN_ORG = "NGAUTOMATE:ACCESS:OWN_ORG"

_bearer = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> CurrentUser:
    try:
        payload = jwt.decode(
            credentials.credentials,
            require_env("SECRET_KEY"),
            algorithms=[ALGORITHM],
            options={"require": ["exp"]},
        )
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")

    user_id = payload.get("user_id")
    org_id = payload.get("org_id")
    if not user_id or not org_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is missing user_id or org_id")

    perms = payload.get("permissions")
    return CurrentUser(
        user_id=str(user_id),
        org_id=str(org_id),
        permissions=frozenset(perms if isinstance(perms, list) else []),
    )


def resolve_org_id(
    target_org_id: str | None = Query(
        None, description="Organization to act for. Defaults to the caller's own org."
    ),
    user: CurrentUser = Depends(get_current_user),
) -> str:
    """The org this request works in.

    - Own org: needs NGAUTOMATE:ACCESS:OWN_ORG or NGAUTOMATE:ACCESS:ALL_ORGS.
    - Another org (target_org_id): needs NGAUTOMATE:ACCESS:ALL_ORGS.
    """
    can_all_orgs = user.has_permission(ACCESS_ALL_ORGS)
    can_own_org = can_all_orgs or user.has_permission(ACCESS_OWN_ORG)

    if not can_own_org:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "No NG Automate access")
    if not target_org_id or target_org_id == user.org_id:
        return user.org_id
    if not can_all_orgs:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "Not allowed to act for another organization"
        )
    return target_org_id
