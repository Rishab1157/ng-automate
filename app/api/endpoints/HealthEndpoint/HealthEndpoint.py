from fastapi import APIRouter

from app.db.mongo import ping

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Is the API up, and can it reach MongoDB?")
async def health() -> dict[str, str]:
    return {"status": "ok", "mongo": "ok" if await ping() else "unreachable"}
