from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path

from app.schemas import AbstractPattern, ReflectionResult, RetrievedExperience, TransformationPlan, ValidationReport


_FORBIDDEN_PATTERNS = [
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"\b\d{6,}\b"),
]


def _safe_lesson(text: str) -> str:
    candidate = " ".join(text.split())[:1200]
    if any(pattern.search(candidate) for pattern in _FORBIDDEN_PATTERNS):
        return "Use the validated abstract strategy and avoid retaining concrete sensitive values."
    return candidate


def _multiset_jaccard(a: dict[str, int], b: dict[str, int]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    intersection = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return intersection / union if union else 1.0


def _set_jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def pattern_similarity(a: AbstractPattern, b: AbstractPattern) -> float:
    score = 0.0
    score += 0.35 * _multiset_jaccard(a.entity_type_counts, b.entity_type_counts)
    score += 0.25 * _set_jaccard(a.relation_types, b.relation_types)
    score += 0.25 * _set_jaccard(a.constraints, b.constraints)
    score += 0.10 * (1.0 if a.task == b.task else 0.0)
    score += 0.05 * (1.0 if a.domain == b.domain else 0.0)
    return max(0.0, min(1.0, score))


class ExperienceMemory:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiences (
                    memory_key TEXT PRIMARY KEY,
                    pattern_fingerprint TEXT NOT NULL,
                    pattern_json TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    tools_json TEXT NOT NULL,
                    result TEXT NOT NULL,
                    failure_type TEXT,
                    outcome_reason TEXT NOT NULL DEFAULT '',
                    lesson TEXT NOT NULL,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    utility_sum REAL NOT NULL DEFAULT 0,
                    observation_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experience_fingerprint ON experiences(pattern_fingerprint)"
            )
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(experiences)").fetchall()
            }
            if "outcome_reason" not in columns:
                conn.execute(
                    "ALTER TABLE experiences ADD COLUMN outcome_reason TEXT NOT NULL DEFAULT ''"
                )
            # Backward-compatible migration from the previous Attempt-2 label.
            conn.execute(
                "UPDATE experiences SET result = 'RELEASED' WHERE result = 'RELEASED_UNVALIDATED'"
            )

    def record(
        self,
        *,
        pattern: AbstractPattern,
        plan: TransformationPlan,
        validation: ValidationReport,
        reflection: ReflectionResult,
        result_override: str | None = None,
    ) -> None:
        # Only abstract structures and validation outcomes are persisted.
        tools = sorted({action.tool for action in plan.actions})
        result = result_override or ("PASS" if validation.accepted else "FAIL")
        if result not in {"PASS", "FAIL", "RELEASED"}:
            raise ValueError(f"Unsupported memory result: {result}")
        failure_type = reflection.failure_type or (validation.violations[0] if validation.violations else None)
        if result == "RELEASED":
            failure_type = None
        outcome_reason = _safe_lesson(reflection.outcome_reason)
        lesson = _safe_lesson(reflection.lesson)
        key_source = "|".join(
            [pattern.fingerprint, plan.strategy, result, failure_type or "", lesson.lower()]
        )
        memory_key = hashlib.sha256(key_source.encode("utf-8")).hexdigest()

        success_inc = 1 if result == "PASS" else 0
        failure_inc = 1 if result == "FAIL" else 0
        utility = float(validation.structural_utility) if validation.validation_performed else 0.0

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiences (
                    memory_key, pattern_fingerprint, pattern_json, strategy, tools_json,
                    result, failure_type, outcome_reason, lesson, success_count, failure_count,
                    utility_sum, observation_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(memory_key) DO UPDATE SET
                    success_count = success_count + excluded.success_count,
                    failure_count = failure_count + excluded.failure_count,
                    utility_sum = utility_sum + excluded.utility_sum,
                    observation_count = observation_count + 1,
                    outcome_reason = excluded.outcome_reason,
                    lesson = excluded.lesson,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    memory_key,
                    pattern.fingerprint,
                    pattern.model_dump_json(),
                    plan.strategy,
                    json.dumps(tools),
                    result,
                    failure_type,
                    outcome_reason,
                    lesson,
                    success_inc,
                    failure_inc,
                    utility,
                ),
            )

    def retrieve(self, pattern: AbstractPattern, *, limit: int = 5) -> list[RetrievedExperience]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiences ORDER BY updated_at DESC LIMIT 500"
            ).fetchall()

        scored: list[tuple[float, sqlite3.Row, AbstractPattern]] = []
        for row in rows:
            stored = AbstractPattern.model_validate_json(row["pattern_json"])
            similarity = pattern_similarity(pattern, stored)
            if similarity < 0.25:
                continue
            evidence = math.log1p(max(1, row["observation_count"])) / math.log(10)
            ranking_score = min(1.0, similarity * (0.9 + 0.1 * min(1.0, evidence)))
            scored.append((ranking_score, row, stored))

        scored.sort(key=lambda item: item[0], reverse=True)
        result: list[RetrievedExperience] = []
        for similarity, row, _stored in scored[:limit]:
            count = max(1, row["observation_count"])
            result.append(
                RetrievedExperience(
                    pattern_fingerprint=row["pattern_fingerprint"],
                    similarity=similarity,
                    strategy=row["strategy"],
                    tools=json.loads(row["tools_json"]),
                    result=row["result"],
                    failure_type=row["failure_type"],
                    outcome_reason=row["outcome_reason"] or row["lesson"],
                    success_count=row["success_count"],
                    failure_count=row["failure_count"],
                    average_utility=row["utility_sum"] / count,
                    lesson=row["lesson"],
                )
            )
        return result

    def stats(self) -> tuple[int, int, int]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(success_count),0) AS s, COALESCE(SUM(failure_count),0) AS f FROM experiences"
            ).fetchone()
        return int(row["n"]), int(row["s"]), int(row["f"])
