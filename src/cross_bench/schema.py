"""Pydantic schemas for serialization, validation, and API boundaries."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class StepRecordSchema(BaseModel):
    step_index: int
    action: str
    input_text: str
    output_text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskResultSchema(BaseModel):
    task_id: str
    run_id: str
    status: str
    steps: list[StepRecordSchema]
    final_output: str
    evaluation: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class LLMConfigSchema(BaseModel):
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 16384


class MemoryConfigSchema(BaseModel):
    provider: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class SystemConfigSchema(BaseModel):
    llm: LLMConfigSchema
    memory: MemoryConfigSchema | None = None
    prompt_strategy_type: str = "zero_shot"


class RunConfigSchema(BaseModel):
    run_id: str
    benchmark: str
    system: SystemConfigSchema
    execution_mode: str = "single_shot"
    max_agent_steps: int = 20
    timeout_seconds: int = 300
    tags: list[str] = Field(default_factory=list)
    created_at: datetime


class RunSummarySchema(BaseModel):
    run_id: str
    benchmark: str
    system: SystemConfigSchema
    execution_mode: str
    status: str
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)
    task_count: int = 0
    pass_count: int = 0
    started_at: datetime
    completed_at: datetime | None = None
    artifact_prefix: str | None = None


class ComparisonSchema(BaseModel):
    """Side-by-side comparison of multiple runs on the same benchmark."""

    benchmark: str
    runs: list[RunSummarySchema]
    metric_columns: list[str]
