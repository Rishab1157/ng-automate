from typing import Any

from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.asynchronous.collection import AsyncCollection

from app.db.mongo import get_db, get_qxcel_db


class BaseRepository:
    collection_name: str = ""
    # Repositories over QXcel's database are read-only: writes raise instead of touching QXcel data.
    use_qxcel_db: bool = False

    @property
    def collection(self) -> AsyncCollection:
        if not self.collection_name:
            raise ValueError("collection_name must be set on the repository")
        database = get_qxcel_db() if self.use_qxcel_db else get_db()
        return database[self.collection_name]

    async def find_by_id(self, id: str, projection: dict[str, Any] | None = None) -> dict[str, Any] | None:
        return await self.collection.find_one({"_id": ObjectId(id)}, projection)

    async def find_one(self, filter: dict[str, Any], projection: dict[str, Any] | None = None) -> dict[str, Any] | None:
        return await self.collection.find_one(filter, projection)

    async def find(
        self,
        filter: dict[str, Any],
        projection: dict[str, Any] | None = None,
        sort: list[tuple[str, int]] | None = None,
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        cursor = self.collection.find(filter, projection)
        if sort:
            cursor = cursor.sort(sort)
        if limit:
            cursor = cursor.limit(limit)
        return await cursor.to_list()

    async def aggregate(self, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cursor = await self.collection.aggregate(pipeline)
        return await cursor.to_list()

    async def insert_one(self, doc: dict[str, Any]) -> str:
        self._ensure_writable()
        result = await self.collection.insert_one(doc)
        return str(result.inserted_id)

    async def update_one(self, filter: dict[str, Any], update: dict[str, Any]) -> int:
        self._ensure_writable()
        result = await self.collection.update_one(filter, update)
        return result.modified_count

    async def find_one_and_update(
        self,
        filter: dict[str, Any],
        update: dict[str, Any],
        projection: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Atomic update that returns the document after the update (None if nothing matched)."""
        self._ensure_writable()
        return await self.collection.find_one_and_update(
            filter, update, projection=projection, return_document=ReturnDocument.AFTER
        )

    def _ensure_writable(self) -> None:
        if self.use_qxcel_db:
            raise PermissionError(f"{type(self).__name__}: QXcel's database is read-only for NG Automate")
