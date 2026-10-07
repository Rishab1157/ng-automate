from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict


class CurrentUserModel(BaseModel):
    """The caller, as described by their QXcel access token."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    org_id: str
    role_id: str | None = None
    email: str = ""
    permissions: frozenset[str] = frozenset()

    def has_permission(self, code: str) -> bool:
        return code in self.permissions

    def has_any_permission(self, codes: Iterable[str]) -> bool:
        return any(code in self.permissions for code in codes)
