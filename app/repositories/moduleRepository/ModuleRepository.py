from app.repositories.baseRepository import BaseRepository


class ModuleRepository(BaseRepository):
    collection_name = "modules"
    use_qxcel_db = True
