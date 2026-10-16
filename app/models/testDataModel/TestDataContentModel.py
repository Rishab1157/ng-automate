from pydantic import BaseModel


class TestDataContentModel(BaseModel):
    """A test-data file's text as it is stored, for the editor. `content` is None (and the file not editable here)
    when the file is too large or not text; `reason` then says why."""

    __test__ = False  # not a pytest test class

    test_data_id: str
    filename: str
    content: str | None = None
    editable: bool
    size_bytes: int
    reason: str | None = None
