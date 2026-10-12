from pydantic import BaseModel, Field


class SaveProjectFileDTO(BaseModel):
    path: str = Field(min_length=1, max_length=1000, description="File path inside the project, e.g. src/test/java/LoginTest.java")
    content: str = Field(description="The whole file as text (UTF-8). A new path creates the file and its folders.")
