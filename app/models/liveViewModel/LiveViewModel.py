from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class LiveViewMode(str, Enum):
    INTERACTIVE = "interactive"  # the viewer may use the mouse and keyboard in the sandbox's browser
    VIEW = "view"  # the viewer only watches


class LiveViewTicketModel(BaseModel):
    """What a live-view ticket opens: one viewer's way into one running test sandbox's display."""

    run_id: str
    org_id: str
    container_id: str
    host_port: int  # websockify of the sandbox, on the host's 127.0.0.1
    vnc_token: str  # websockify's token for the VNC server (interactive or view-only); never sent to the browser
    mode: LiveViewMode
    expires_at: datetime
