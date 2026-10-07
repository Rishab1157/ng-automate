from pydantic import BaseModel, SecretStr


class ModelConnectionSourceModel(BaseModel):
    """A QXcel model connection joined with its provider and model type. Internal only: never returned by the API."""

    id: str
    userdefined_name: str
    provider_code: str
    api_base_url: str | None = None
    # QXcel model_types.type_code, e.g. "gpt-4o", "GEMINI-2.5-PRO"
    model_name: str
    # SecretStr: hidden from print, repr and logs.
    api_key: SecretStr | None = None
