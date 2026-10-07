import logging

logger = logging.getLogger(__name__)

class DiscardOnError:
    def __init__(self, storage, project_id):
        self.storage = storage
        self.project_id = project_id

    def __enter__(self):
        logger.info(f"Entering {type(self).__name__} context")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.error(
                "Exception occurred: %s: %s",
                exc_type.__name__,
                exc_value,
                exc_info=(exc_type, exc_value, traceback),
            )

            self.storage.delete(str(self.project_id))

        return False