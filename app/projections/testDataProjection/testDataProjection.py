# Lists leave out the raw sources (up to 100 KB each).
TEST_DATA_LIST_PROJECTION = {
    "_id": 1,
    "org_id": 1,
    "project_id": 1,
    "created_by": 1,
    "filename": 1,
    "source_format": 1,
    "status": 1,
    "model_connection_id": 1,
    "data_set": 1,
    "case_count": 1,
    "error": 1,
    "created_at": 1,
    "updated_at": 1,
}

TEST_DATA_DETAIL_PROJECTION = {
    "_id": 1,
    "org_id": 1,
    "project_id": 1,
    "created_by": 1,
    "filename": 1,
    "source_format": 1,
    "status": 1,
    "raw_source": 1,
    "model_connection_id": 1,
    "data_set": 1,
    "case_count": 1,
    "error": 1,
    "created_at": 1,
    "updated_at": 1,
}
