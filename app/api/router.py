from fastapi import APIRouter

from app.api.endpoints import HealthEndpoint, LiveViewEndpoint, ProjectEndpoint, RunEndpoint

api_router = APIRouter()

api_router.include_router(HealthEndpoint.router)
api_router.include_router(ProjectEndpoint.router)
api_router.include_router(RunEndpoint.router)
api_router.include_router(RunEndpoint.socket_router)
api_router.include_router(LiveViewEndpoint.router)
api_router.include_router(LiveViewEndpoint.viewer_router)
