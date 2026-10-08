# Everything a RunModel holds, including the stage outputs a resumed run needs. Leaves out the event counter.
RUN_DETAIL_PROJECTION = {
    "_id": 1,
    "org_id": 1,
    "project_id": 1,
    "created_by": 1,
    "model_connection_id": 1,
    "mode": 1,
    "test_selector": 1,
    "test_data_id": 1,
    "run_scope": 1,
    "status": 1,
    "stage": 1,
    "outputs": 1,
    "sandbox_container_id": 1,
    "error": 1,
    "created_at": 1,
    "updated_at": 1,
    "started_at": 1,
    "finished_at": 1,
}

RUN_EVENT_PROJECTION = {
    "_id": 0,
    "run_id": 1,
    "seq": 1,
    "type": 1,
    "level": 1,
    "message": 1,
    "data": 1,
    "created_at": 1,
}
