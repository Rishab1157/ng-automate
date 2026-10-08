import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.agents.MasterAgent.RunExecutor import run_executor
from app.api.router import api_router
from app.config import settings
from app.core.exceptions import (
    NgAutomateException,
    generic_exception_handler,
    http_exception_handler,
    ng_automate_exception_handler,
    validation_exception_handler,
)
from app.core.logging_config import setup_logging
from app.db.indexes import ensure_indexes
from app.db.mongo import close_client, ping
from app.db.qdrant import close_qdrant_client
from app.services.testDataService import test_data_processor

logger = logging.getLogger(__name__)

DESCRIPTION = """
Agent system for test-automation projects.

**Auth:** click **Authorize** and paste a QXcel access token (without the word `Bearer`).
Endpoints work in your own organization; `?target_org_id=` needs `NGAUTOMATE:ACCESS:ALL_ORGS`.
"""


def _load_agent_libraries() -> None:
    """The agents' libraries (OpenHands, LiteLLM, LangGraph) take ~10 s to import. Loading them in the background
    right after startup keeps the API responsive and the first run from stalling the event loop."""
    try:
        import app.agents.MasterAgent.MasterGraph  # noqa: F401
        import app.agents.MasterAgent.MasterNodes  # noqa: F401
        import app.agents.TestGeneratorAgent.TestCaseExtractor  # noqa: F401
    except Exception:
        logger.exception("Could not preload the agent libraries; they load on first use instead")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    preload = asyncio.get_running_loop().run_in_executor(None, _load_agent_libraries)
    if await ping():
        await ensure_indexes()
        try:
            resumed = await run_executor.resume_interrupted()
            logger.info("Resumed %d interrupted run(s)", resumed)
            reading = await test_data_processor.resume_processing()
            if reading:
                logger.info("Reading %d test-data source(s) again", reading)
        except Exception:
            # The API still starts; the runs stay unfinished and are resumed on the next start.
            logger.exception("Could not resume interrupted runs")
    else:
        logger.warning("MongoDB not reachable at startup (%s)", settings.MONGO_DATABASE)
    yield
    await preload
    # Running runs stop unfinished (they resume on the next start) before the database client goes away.
    await run_executor.shutdown()
    await test_data_processor.shutdown()
    await close_client()
    await close_qdrant_client()


app = FastAPI(
    lifespan=lifespan,
    title="NG Automate",
    description=DESCRIPTION,
    version="0.1.0",
    # Keep the pasted token after a page refresh.
    swagger_ui_parameters={"persistAuthorization": True},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    # Auth is a Bearer header, not cookies, so credentials stay off.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

app.add_exception_handler(NgAutomateException, ng_automate_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)
