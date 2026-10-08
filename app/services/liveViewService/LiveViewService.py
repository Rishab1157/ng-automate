"""The live view: users watch, and can take over, the browser a run's tests drive in its test sandbox.

The sandbox shows its display through websockify on the host's 127.0.0.1 (never published to the network). A user
with access to the run asks for a ticket; the ticket opens the noVNC page and its WebSocket through our API, which
browsers can load in an iframe without an Authorization header. The ticket is random, short-lived for new
connections, stored only as a hash, and bound to one sandbox; the VNC token behind it never reaches the browser.
"""

import asyncio
import logging
import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.config import settings
from app.core.exceptions import ErrorMessages, LiveViewUnavailableError
from app.models.liveViewModel import LiveViewMapper, LiveViewMode, LiveViewTicketModel, ticket_hash
from app.models.runModel import RunStatus
from app.projections.liveViewProjection import LIVE_VIEW_TICKET_PROJECTION
from app.repositories.liveViewTicketRepository import LiveViewTicketRepository
from app.services.runService import RunService

if TYPE_CHECKING:
    # The sandbox service pulls in the OpenHands SDK (seconds to import): the API loads it on first use only.
    from app.services.sandboxService import SandboxService

logger = logging.getLogger(__name__)


class LiveViewService:
    def __init__(self, sandbox_service: "SandboxService | None" = None) -> None:
        self.ticket_repo = LiveViewTicketRepository()
        self.run_service = RunService()
        self._sandbox_service = sandbox_service

    @property
    def sandbox_service(self) -> "SandboxService":
        if self._sandbox_service is None:
            from app.services.sandboxService import SandboxService

            self._sandbox_service = SandboxService()
        return self._sandbox_service

    async def create_ticket(
        self, *, run_id: str, org_id: str, user_id: str, mode: LiveViewMode
    ) -> tuple[str, datetime]:
        """A ticket for the run's current test sandbox, and when it stops opening new connections."""
        run = await self.run_service.get(run_id, org_id)
        container_id = run.sandbox_container_id
        if run.status != RunStatus.RUNNING or not container_id:
            raise LiveViewUnavailableError(ErrorMessages.LIVE_VIEW_NO_SANDBOX)
        host_port = await asyncio.to_thread(self.sandbox_service.live_view_port, container_id)
        if host_port is None:
            # e.g. the analyzer's read-only sandbox: it has no browser and no display.
            raise LiveViewUnavailableError(ErrorMessages.LIVE_VIEW_NO_SANDBOX)
        from app.services.sandboxService.SandboxService import LiveViewError

        try:
            vnc_token = await asyncio.to_thread(
                self.sandbox_service.add_live_view_token, container_id, mode == LiveViewMode.INTERACTIVE
            )
        except LiveViewError as error:
            logger.warning("No live view for run %s: %s", run_id, error)
            raise LiveViewUnavailableError(ErrorMessages.LIVE_VIEW_NO_SANDBOX) from None

        ticket = secrets.token_urlsafe(32)
        db_model = LiveViewMapper.to_create_db_model(
            ticket=ticket,
            run_id=run_id,
            org_id=org_id,
            user_id=user_id,
            container_id=container_id,
            host_port=host_port,
            vnc_token=vnc_token,
            mode=mode,
            lifetime_seconds=settings.LIVE_VIEW_TICKET_SECONDS,
        )
        await self.ticket_repo.insert_one(db_model.model_dump(by_alias=True))
        logger.info("Live view ticket for run %s (%s)", run_id, mode.value)
        return ticket, db_model.expires_at

    async def resolve(self, ticket: str) -> LiveViewTicketModel | None:
        """What a ticket opens, or None when it is unknown or expired."""
        if not ticket or len(ticket) > 200:
            return None
        doc = await self.ticket_repo.find_one({"_id": ticket_hash(ticket)}, LIVE_VIEW_TICKET_PROJECTION)
        if doc is None:
            return None
        model = LiveViewMapper.to_model(doc)
        # The TTL monitor runs about once a minute: check the time as well.
        return model if model.expires_at > datetime.now(UTC) else None
