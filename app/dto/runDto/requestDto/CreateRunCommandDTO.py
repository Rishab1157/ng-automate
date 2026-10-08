from pydantic import BaseModel, Field

from app.models.runCommandModel import RunCommandType


class CreateRunCommandDTO(BaseModel):
    type: RunCommandType = Field(
        description="message: tell the working agent something (text). pause / resume: the working agent. "
        "stop: end the run."
    )
    text: str | None = Field(None, max_length=4000, description="The message (type message only)")
