from typing import Any

from bson import ObjectId
from pydantic import field_validator


def object_id_validator(field: str) -> Any:
    """Reusable validator: the field must be a valid MongoDB ObjectId string."""

    def validate(cls: type, value: str) -> str:
        if not ObjectId.is_valid(value):
            raise ValueError(f"{field} is not a valid id")
        return value

    return field_validator(field)(validate)
