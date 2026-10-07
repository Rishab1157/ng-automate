from typing import Any

from pydantic import SecretStr

from .GitConnectionModel import GitCloneSourceModel


class GitConnectionMapper:
    @staticmethod
    def to_clone_source_model(doc: dict[str, Any]) -> GitCloneSourceModel:
        """doc: a git_connections document joined with its git_providers entry as `provider`."""
        provider = doc.get("provider") or {}
        token = doc.get("token")
        return GitCloneSourceModel(
            id=str(doc["_id"]),
            userdefined_name=doc.get("userdefined_name") or "",
            repo_url=doc["repo_url"],
            branch=doc.get("branch") or "main",
            provider_code=doc.get("provider_code") or provider.get("provider_code"),
            token=SecretStr(token) if token else None,
        )
