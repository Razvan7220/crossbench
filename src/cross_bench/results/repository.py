"""Postgres-backed results repository with SQLAlchemy async ORM."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from cross_bench.models import RunConfig, TaskResult


class Base(DeclarativeBase):
    pass


# ``run_id`` is a String, not a UUID. Per workflow.md ("Run IDs encode
# intent"), human-readable parents like ``locomo-mem0v2-20260423-120000``
# are first-class — the dispatcher's UUID5 shard IDs are merely one
# concrete shape that fits. Existing Postgres deployments with the legacy
# UUID column need ``ALTER TABLE runs ALTER COLUMN run_id TYPE TEXT;`` and
# the matching ``ALTER`` on ``task_results.run_id``.
class RunRow(Base):
    __tablename__ = "runs"

    run_id = Column(String, primary_key=True)
    benchmark = Column(String, nullable=False, index=True)
    system_config = Column(JSONB, nullable=False)
    execution_mode = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, index=True)
    completed_at = Column(DateTime(timezone=True))
    aggregate_metrics = Column(JSONB)
    artifact_prefix = Column(String)
    status = Column(String, default="pending")
    tags = Column(JSONB, default=list)


class TaskResultRow(Base):
    __tablename__ = "task_results"
    __table_args__ = (UniqueConstraint("run_id", "task_id"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, index=True)
    task_id = Column(String, nullable=False)
    status = Column(String, nullable=False)
    final_output = Column(Text)
    evaluation = Column(JSONB)
    step_count = Column(Integer)
    total_tokens = Column(Integer)
    total_latency_ms = Column(Float)
    error = Column(Text)


class ResultRepository:
    def __init__(self, database_url: str):
        self._engine = create_async_engine(database_url, echo=False)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init_db(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def create_run(self, config: RunConfig) -> None:
        async with self._session_factory() as session:
            row = RunRow(
                run_id=config.run_id,
                benchmark=config.benchmark,
                system_config=_safe_json(asdict(config.system)),
                execution_mode=config.execution_mode.value,
                created_at=config.created_at,
                status="running",
                tags=config.tags,
            )
            session.add(row)
            await session.commit()

    async def save_task_result(self, result: TaskResult) -> None:
        async with self._session_factory() as session:
            total_tokens = sum(s.tokens_in + s.tokens_out for s in result.steps)
            total_latency = sum(s.latency_ms for s in result.steps)
            row = TaskResultRow(
                run_id=result.run_id,
                task_id=result.task_id,
                status=result.status.value,
                final_output=result.final_output[:10000],
                evaluation=result.evaluation,
                step_count=len(result.steps),
                total_tokens=total_tokens,
                total_latency_ms=total_latency,
                error=result.error,
            )
            session.add(row)
            await session.commit()

    async def complete_run(
        self, run_id: str, metrics: dict[str, float], completed_at: datetime
    ) -> None:
        async with self._session_factory() as session:
            from sqlalchemy import update
            await session.execute(
                update(RunRow)
                .where(RunRow.run_id == run_id)
                .values(
                    aggregate_metrics=metrics,
                    completed_at=completed_at,
                    status="completed",
                )
            )
            await session.commit()

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        async with self._session_factory() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(RunRow).where(RunRow.run_id == run_id)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return {
                "run_id": row.run_id,
                "benchmark": row.benchmark,
                "system_config": row.system_config,
                "execution_mode": row.execution_mode,
                "created_at": row.created_at,
                "completed_at": row.completed_at,
                "aggregate_metrics": row.aggregate_metrics,
                "status": row.status,
                "tags": row.tags,
            }

    async def list_runs(
        self, benchmark: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            from sqlalchemy import select
            stmt = select(RunRow).order_by(RunRow.created_at.desc()).limit(limit)
            if benchmark:
                stmt = stmt.where(RunRow.benchmark == benchmark)
            result = await session.execute(stmt)
            return [
                {
                    "run_id": r.run_id,
                    "benchmark": r.benchmark,
                    "execution_mode": r.execution_mode,
                    "status": r.status,
                    "aggregate_metrics": r.aggregate_metrics,
                    "created_at": r.created_at,
                    "completed_at": r.completed_at,
                }
                for r in result.scalars()
            ]

    async def get_task_results(self, run_id: str) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(TaskResultRow).where(TaskResultRow.run_id == run_id)
            )
            return [
                {
                    "task_id": r.task_id,
                    "status": r.status,
                    "evaluation": r.evaluation,
                    "step_count": r.step_count,
                    "total_tokens": r.total_tokens,
                    "total_latency_ms": r.total_latency_ms,
                    "error": r.error,
                }
                for r in result.scalars()
            ]

    async def compare_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch multiple runs for side-by-side comparison."""
        runs = []
        for rid in run_ids:
            run = await self.get_run(rid)
            if run:
                runs.append(run)
        return runs


class NullResultRepository:
    """No-op repository for local runs without a database."""

    async def init_db(self) -> None:
        pass

    async def create_run(self, config: RunConfig) -> None:
        pass

    async def save_task_result(self, result: TaskResult) -> None:
        pass

    async def complete_run(
        self, run_id: str, metrics: dict[str, float], completed_at: datetime
    ) -> None:
        pass

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        return None

    async def list_runs(
        self, benchmark: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        return []

    async def get_task_results(self, run_id: str) -> list[dict[str, Any]]:
        return []

    async def compare_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        return []


def _safe_json(obj: Any) -> Any:
    """Ensure object is JSON-serializable for JSONB columns."""
    return json.loads(json.dumps(obj, default=str))
