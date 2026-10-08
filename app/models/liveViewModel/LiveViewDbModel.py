from datetime import datetime

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field


class LiveViewTicketCreateDbModel(BaseModel):
    """A new document in `live_view_tickets`. Its id is the SHA-256 of the ticket: the ticket itself is never stored."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    id: str = Field(alias="_id")
    run_id: ObjectId
    org_id: ObjectId
    created_by: ObjectId
    container_id: str
    host_port: int
    vnc_token: str
    mode: str
    created_at: datetime
    # A TTL index removes the document after this time.
    expires_at: datetime
