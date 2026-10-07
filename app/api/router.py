from fastapi import APIRouter

from app.api.endpoints import HealthEndpoint, ProjectEndpoint, RunEndpoint

api_router = APIRouter()

api_router.include_router(HealthEndpoint.router)
api_router.include_router(ProjectEndpoint.router)
api_router.include_router(RunEndpoint.router)
