from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


Sensitivity = Literal["low", "medium", "high", "critical"]


class AnalyzedEntity(BaseModel):
    id: str
    type: str
    role: str = "unknown"
    value: str = Field(min_length=1)
    sensitive: bool = True
    sensitivity: Sensitivity = "high"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    source: Literal["llm", "backstop", "coverage", "residual"] = "llm"

    @model_validator(mode="after")
    def validate_offsets(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must either both be present or both be absent")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class SemanticRelation(BaseModel):
    source_id: str
    relation: str
    target_id: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class AnalysisResult(BaseModel):
    task: str = "privacy_preserving_transformation"
    domain: str = "general"
    entities: list[AnalyzedEntity] = Field(default_factory=list)
    relations: list[SemanticRelation] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class AbstractPattern(BaseModel):
    task: str
    domain: str
    entity_types: list[str]
    semantic_roles: list[str]
    relation_types: list[str]
    constraints: list[str]
    entity_type_counts: dict[str, int]
    fingerprint: str


class RetrievedExperience(BaseModel):
    pattern_fingerprint: str
    similarity: float = Field(ge=0.0, le=1.0)
    strategy: str
    tools: list[str]
    result: Literal["PASS", "FAIL", "RELEASED"]
    failure_type: str | None = None
    outcome_reason: str
    success_count: int = 0
    failure_count: int = 0
    average_utility: float = 0.0
    lesson: str


class PlanAction(BaseModel):
    target_ids: list[str] = Field(min_length=1)
    tool: str
    rationale: str = ""


class TransformationPlan(BaseModel):
    strategy: str
    actions: list[PlanAction]
    required_constraints: list[str]
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    rationale: str = ""


class ReplacementRecord(BaseModel):
    entity_id: str
    original_start: int
    original_end: int
    replacement_start: int
    replacement_end: int
    replacement: str


class ExecutionResult(BaseModel):
    candidate_text: str
    replacements: list[ReplacementRecord]
    entity_replacements: dict[str, str]


class ValidationReport(BaseModel):
    accepted: bool
    validation_performed: bool = True
    privacy_pass: bool
    privacy_risk: float = Field(ge=0.0, le=1.0)
    structural_utility: float = Field(ge=0.0, le=1.0)
    numeric_consistency: float = Field(ge=0.0, le=1.0)
    temporal_consistency: float = Field(ge=0.0, le=1.0)
    relation_consistency: float = Field(ge=0.0, le=1.0)
    violations: list[str] = Field(default_factory=list)


class ReflectionResult(BaseModel):
    outcome_reason: str
    lesson: str
    failure_type: str | None = None
    recommended_strategy: str | None = None
    recommended_tools: list[str] = Field(default_factory=list)


class AttemptSummary(BaseModel):
    attempt: int = Field(ge=1)
    outcome: Literal["PASS", "FAIL", "RELEASED"] = "FAIL"
    released: bool = False
    strategy: str | None = None
    repaired: bool = False
    coverage_repaired: bool = False
    failure_class: str | None = None
    new_entity_types: list[str] = Field(default_factory=list)
    accepted: bool
    violations: list[str] = Field(default_factory=list)
    outcome_reason: str = ""
    learned_lesson: str = ""


class ProcessRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)
    scope: str = Field(default="default", min_length=1, max_length=128)


class ProcessResponse(BaseModel):
    accepted: bool
    transformed_text: str | None
    attempts: int
    pattern: AbstractPattern | None
    strategy: str | None
    validation: ValidationReport | None
    attempt_history: list[AttemptSummary] = Field(default_factory=list)
    message: str


class MemoryStats(BaseModel):
    total_records: int
    total_successes: int
    total_failures: int
