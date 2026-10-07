from bson import ObjectId

from app.core.exceptions import ErrorMessages, NotFoundError
from app.repositories.moduleRepository import ModuleRepository

NGAUTOMATE_MODULE_CODE = "NGAUTOMATE"


class ModuleService:
    """Looks up QXcel modules (crews). Ids differ per environment, so they are found by code."""

    # Shared by all instances on purpose: module ids never change while the app runs.
    _module_ids: dict[str, ObjectId] = {}

    def __init__(self) -> None:
        self.module_repo = ModuleRepository()

    async def get_module_id(self, module_code: str) -> ObjectId:
        if module_code not in self._module_ids:
            doc = await self.module_repo.find_one({"module_code": module_code}, {"_id": 1})
            if doc is None:
                raise NotFoundError(ErrorMessages.MODULE_NOT_FOUND.format(code=module_code))
            self._module_ids[module_code] = doc["_id"]
        return self._module_ids[module_code]
