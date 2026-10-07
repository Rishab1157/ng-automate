"""STUB (interface only) — replaced by the master build step."""


class RunExecutor:
    async def submit(self, run_id: str) -> None:
        raise NotImplementedError

    async def resume_interrupted(self) -> int:
        raise NotImplementedError

    async def shutdown(self) -> None:
        raise NotImplementedError


run_executor = RunExecutor()
