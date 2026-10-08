from app.repositories.baseRepository import BaseRepository


class TestDataRepository(BaseRepository):
    __test__ = False  # not a pytest test class

    collection_name = "test_data"
