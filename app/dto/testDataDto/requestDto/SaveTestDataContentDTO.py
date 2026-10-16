from pydantic import BaseModel


class SaveTestDataContentDTO(BaseModel):
    """A test-data file's new text, from the editor. It is read again like an upload (same checks and limits)."""

    content: str
