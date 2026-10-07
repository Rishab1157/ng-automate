from app.repositories.baseRepository import BaseRepository


class ModelConnectionRepository(BaseRepository):
    collection_name = "model_connections"
    use_qxcel_db = True
