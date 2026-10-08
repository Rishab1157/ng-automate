from pymongo import ASCENDING, DESCENDING

from app.repositories.liveViewTicketRepository import LiveViewTicketRepository
from app.repositories.projectProfileRepository import ProjectProfileRepository
from app.repositories.runCommandRepository import RunCommandRepository
from app.repositories.projectRepository import ProjectRepository
from app.repositories.runEventRepository import RunEventRepository
from app.repositories.runRepository import RunRepository
from app.repositories.testDataRepository import TestDataRepository


async def ensure_indexes() -> None:
    """Create our indexes if missing. Safe to run on every startup."""
    await ProjectRepository().collection.create_index(
        [("org_id", ASCENDING), ("created_at", DESCENDING)], name="org_id_created_at"
    )
    await RunRepository().collection.create_index(
        [("org_id", ASCENDING), ("created_at", DESCENDING)], name="org_id_created_at"
    )
    # Startup looks up unfinished runs to resume them.
    await RunRepository().collection.create_index([("status", ASCENDING)], name="status")
    # Events are read in order; (run_id, seq) also guarantees no duplicate sequence numbers.
    await RunEventRepository().collection.create_index(
        [("run_id", ASCENDING), ("seq", ASCENDING)], name="run_id_seq", unique=True
    )
    await ProjectProfileRepository().collection.create_index(
        [("project_id", ASCENDING), ("created_at", DESCENDING)], name="project_id_created_at"
    )
    # Workers read the pending commands of their run, oldest first.
    await RunCommandRepository().collection.create_index(
        [("run_id", ASCENDING), ("status", ASCENDING), ("_id", ASCENDING)], name="run_id_status_id"
    )
    # Live-view tickets disappear once they expire.
    await LiveViewTicketRepository().collection.create_index("expires_at", name="expires_at_ttl", expireAfterSeconds=0)
    await TestDataRepository().collection.create_index(
        [("project_id", ASCENDING), ("created_at", DESCENDING)], name="project_id_created_at"
    )
