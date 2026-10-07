from dataclasses import dataclass


@dataclass(frozen=True)
class CurrentUser:
    """The caller, read from the QXcel JWT."""

    user_id: str
    org_id: str
    permissions: frozenset[str]

    def has_permission(self, code: str) -> bool:
        return code in self.permissions
