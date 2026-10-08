from datetime import datetime

from pydantic import BaseModel

from app.models.liveViewModel import LiveViewMode


class LiveViewResponseDTO(BaseModel):
    """Open `viewer_url` (relative to the API) in an iframe or a new tab: it needs no Authorization header."""

    viewer_url: str
    mode: LiveViewMode
    # New connections need the ticket before this time; an open connection stays open.
    expires_at: datetime
