import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

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

logger = logging.getLogger(__name__)

DESCRIPTION = """
Agent system for test-automation projects.

**Auth:** click **Authorize** and paste a QXcel access token (without the word `Bearer`).
Endpoints work in your own organization; `?target_org_id=` needs `NGAUTOMATE:ACCESS:ALL_ORGS`.
"""


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    if await ping():
        await ensure_indexes()
    else:
        logger.warning("MongoDB not reachable at startup (%s)", settings.MONGO_DATABASE)
    yield
    await close_client()


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
