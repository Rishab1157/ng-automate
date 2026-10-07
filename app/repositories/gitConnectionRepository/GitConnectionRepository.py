from app.repositories.baseRepository import BaseRepository


class GitConnectionRepository(BaseRepository):
    collection_name = "git_connections"
    use_qxcel_db = True
