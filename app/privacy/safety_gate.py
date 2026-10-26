from __future__ import annotations

import re
from decimal import Decimal
from difflib import SequenceMatcher
from itertools import combinations

from dateutil import parser as date_parser

from app.schemas import AnalysisResult, ExecutionResult, TransformationPlan, ValidationReport
from app.tools.registry import DeterministicToolRegistry, _parse_number
from app.privacy.relation_graph import date_relation_groups, numeric_relation_groups, person_alias_groups


_GENERIC_FRAGMENTS = {
    "example", "street", "avenue", "road", "drive", "company", "organization",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "gmail", "yahoo", "outlook",
    # Generic credential/technical vocabulary is structural, not user-specific.
    # Treating these words as leaked fragments causes false collisions for safe
    # synthetic credentials such as ``access_test_...``.
    "access", "token", "credential", "password", "secret", "private",
    "connection", "database", "postgresql", "synthetic", "account",
    "device", "identifier", "employee", "customer", "passport", "driver",
    "routing", "credit", "medical", "patient", "policy", "claim", "student",
    "username", "sample", "record",
}

_MONEY_TYPES = {"MONEY", "SALARY", "INCOME", "FINANCIAL_VALUE"}
_NUMERIC_TYPES = _MONEY_TYPES | {"AGE", "COUNT", "PERCENTAGE", "NUMBER"}
_NUMERIC_CONSTRAINTS = {
    "PRESERVE_NUMERIC_DIFFERENCE",
    "PRESERVE_CHANGE_DIRECTION",
    # Backward compatibility for analyses/memory created by older versions.
    "PRESERVE_NUMERIC_RATIO",
    "PRESERVE_PERCENT_CHANGE",
}
_DATE_TYPES = {"DATE", "ISO_DATE", "DOB", "DATETIME"}
_FORBIDDEN_PLACEHOLDER_TOKENS = (
    "REDACTED",
    "MASKED",
    "HIDDEN",
    "REMOVED",
)


def _norm(s: str) -> str:
    return s.strip().upper().replace(" ", "_").replace("-", "_")


def _looks_code_like(value: str) -> bool:
    stripped = value.strip()
    compact = re.sub(r"[^A-Za-z0-9]", "", stripped)
    return (
        bool(compact)
        and any(ch.isdigit() for ch in compact)
        and ("-" in stripped or "_" in stripped or len(compact) >= 12)
        and len(stripped.split()) <= 3
    )


def _normalized_value(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _compact_alnum(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _fragments(value: str) -> set[str]:
    parts = set(re.findall(r"[A-Za-z]{5,}|\d{5,}", value.lower()))
    return {p for p in parts if p not in _GENERIC_FRAGMENTS}


def _is_forbidden_placeholder(value: str) -> bool:
    normalized = value.strip().upper()
    return any(token in normalized for token in _FORBIDDEN_PLACEHOLDER_TOKENS)


def _skeleton_original(text: str, analysis: AnalysisResult) -> str:
    pieces: list[str] = []
    cursor = 0
    entities = sorted(
        (e for e in analysis.entities if e.sensitive),
        key=lambda e: e.start if e.start is not None else -1,
    )
    for entity in entities:
        if entity.start is None or entity.end is None:
            raise ValueError(f"Missing aligned offsets for {entity.id}")
        pieces.append(text[cursor:entity.start])
        pieces.append(f"<{_norm(entity.type)}>")
        cursor = entity.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _skeleton_candidate(candidate: str, analysis: AnalysisResult, execution: ExecutionResult) -> str:
    types = {e.id: _norm(e.type) for e in analysis.entities if e.sensitive}
    pieces: list[str] = []
    cursor = 0
    records = sorted(execution.replacements, key=lambda r: r.replacement_start)
    for record in records:
        pieces.append(candidate[cursor:record.replacement_start])
        pieces.append(f"<{types.get(record.entity_id, 'UNKNOWN_SENSITIVE')}>")
        cursor = record.replacement_end
    pieces.append(candidate[cursor:])
    return "".join(pieces)


class HardSafetyGate:
    """Deterministic, non-bypassable release gate for synthetic-only output."""

    def __init__(self, registry: DeterministicToolRegistry):
        self.registry = registry

    def validate_plan(self, analysis: AnalysisResult, plan: TransformationPlan) -> list[str]:
        violations: list[str] = []
        entities = {e.id: e for e in analysis.entities if e.sensitive}
        seen: dict[str, int] = {entity_id: 0 for entity_id in entities}

        analysis_constraints = {_norm(c) for c in analysis.constraints}
        plan_constraints = {_norm(c) for c in plan.required_constraints}
        missing_constraints = analysis_constraints - plan_constraints
        if missing_constraints:
            violations.append(
                "PLAN_OMITS_REQUIRED_CONSTRAINTS:" + ",".join(sorted(missing_constraints))
            )

        for action in plan.actions:
            if action.tool in {"suppress_secret", "generalize_number"}:
                violations.append(f"PLAN_USES_FORBIDDEN_TOOL:{action.tool}")

            for target_id in action.target_ids:
                if target_id not in entities:
                    violations.append(f"PLAN_UNKNOWN_TARGET:{target_id}")
                    continue
                seen[target_id] += 1
                entity = entities[target_id]
                normalized_type = _norm(entity.type)
                allowed = self.registry.allowed_tools_for(normalized_type)
                if action.tool not in allowed:
                    violations.append(f"PLAN_DISALLOWED_TOOL:{target_id}:{action.tool}")

                semantic_types = {
                    "HEALTH_INFORMATION",
                    "DEMOGRAPHIC_SENSITIVE",
                    "EMPLOYMENT_INFORMATION",
                    "EDUCATION_INFORMATION",
                    "LEGAL_INFORMATION",
                    "INSURANCE_INFORMATION",
                    "BUSINESS_CONFIDENTIAL",
                    "RESEARCH_CONFIDENTIAL",
                }
                if normalized_type in semantic_types and action.tool != "synthetic_context":
                    violations.append(f"PLAN_CONTEXT_LOSS_TOOL:{target_id}:{action.tool}")
                elif normalized_type in {"QUASI_IDENTIFIER", "UNKNOWN_SENSITIVE"}:
                    if not _looks_code_like(entity.value) and action.tool == "synthetic_identifier":
                        violations.append(f"PLAN_CONTEXT_LOSS_TOOL:{target_id}:{action.tool}")

        for entity_id, count in seen.items():
            if count == 0:
                violations.append(f"PLAN_UNHANDLED_ENTITY:{entity_id}")
            elif count > 1:
                violations.append(f"PLAN_DUPLICATE_ENTITY:{entity_id}")

        action_by_target: dict[str, tuple[int, str]] = {}
        for idx, action in enumerate(plan.actions):
            for target in action.target_ids:
                action_by_target[target] = (idx, action.tool)

        for group in person_alias_groups(analysis):
            member_ids = sorted(group)
            action_refs = [action_by_target.get(entity_id) for entity_id in member_ids]
            if (
                any(ref is None for ref in action_refs)
                or len({ref[0] for ref in action_refs if ref is not None}) != 1
                or any(ref is None or ref[1] != "synthetic_identity" for ref in action_refs)
            ):
                violations.append(
                    "PERSON_ALIASES_NOT_GROUPED:" + ":".join(member_ids)
                )

        numeric_relation_required = bool(analysis_constraints & _NUMERIC_CONSTRAINTS)
        temporal_relation_required = bool(
            analysis_constraints & {"PRESERVE_DATE_INTERVAL", "PRESERVE_DATE_ORDER"}
        )

        if numeric_relation_required:
            for group in numeric_relation_groups(analysis):
                member_ids = sorted(group)
                action_refs = [action_by_target.get(entity_id) for entity_id in member_ids]
                if (
                    any(ref is None for ref in action_refs)
                    or len({ref[0] for ref in action_refs if ref is not None}) != 1
                    or any(
                        ref is None or ref[1] not in {"shared_numeric_shift", "proportional_money"}
                        for ref in action_refs
                    )
                ):
                    violations.append(
                        "NUMERIC_ENTITIES_NOT_GROUPED:" + ":".join(member_ids)
                    )

        if temporal_relation_required:
            for group in date_relation_groups(analysis):
                date_ids = sorted(group)
                action_refs = [action_by_target.get(entity_id) for entity_id in date_ids]
                if (
                    any(ref is None for ref in action_refs)
                    or len({ref[0] for ref in action_refs if ref is not None}) != 1
                    or any(ref is None or ref[1] != "shared_date_shift" for ref in action_refs)
                ):
                    violations.append(
                        "DATE_ENTITIES_NOT_GROUPED:" + ":".join(date_ids)
                    )

        return sorted(set(violations))

    def validate(
        self,
        *,
        original_text: str,
        analysis: AnalysisResult,
        plan: TransformationPlan,
        execution: ExecutionResult,
    ) -> ValidationReport:
        violations = self.validate_plan(analysis, plan)
        candidate = execution.candidate_text
        replacements = execution.entity_replacements
        sensitive_entities = [e for e in analysis.entities if e.sensitive]

        # Every sensitive entity must receive a non-empty synthetic replacement.
        for entity in sensitive_entities:
            if entity.id not in replacements:
                violations.append(f"MISSING_SYNTHETIC_REPLACEMENT:{entity.id}")
                continue
            replacement = replacements[entity.id]
            if not replacement.strip():
                violations.append(f"EMPTY_SYNTHETIC_REPLACEMENT:{entity.id}")
            if _is_forbidden_placeholder(replacement):
                violations.append(f"FORBIDDEN_PLACEHOLDER:{entity.id}")

        # Exact and partial source leakage checks.
        normalized_candidate = _normalized_value(candidate)
        compact_candidate = _compact_alnum(candidate)

        for entity in sensitive_entities:
            original = entity.value
            normalized_original = _normalized_value(original)
            if normalized_original and normalized_original in normalized_candidate:
                violations.append(f"EXACT_LEAK:{entity.id}")
                continue

            entity_type = _norm(entity.type)
            if entity_type not in _NUMERIC_TYPES | _DATE_TYPES:
                compact_original = _compact_alnum(original)
                if len(compact_original) >= 8 and compact_original in compact_candidate:
                    violations.append(f"NORMALIZED_LEAK:{entity.id}")
                    continue
                replacement = replacements.get(entity.id, "")
                replacement_lower = replacement.lower()
                for fragment in _fragments(original):
                    if fragment in replacement_lower:
                        violations.append(f"PARTIAL_LEAK:{entity.id}")
                        break

        # Known-safe namespaces for structured synthetic values.
        for entity in sensitive_entities:
            if entity.id not in replacements:
                continue
            replacement = replacements[entity.id]
            entity_type = _norm(entity.type)

            if entity_type == "EMAIL" and not replacement.lower().endswith("@example.com"):
                violations.append(f"UNSAFE_EMAIL_SURROGATE:{entity.id}")
            elif entity_type == "PHONE" and not replacement.startswith("+1 202-555-01"):
                violations.append(f"UNSAFE_PHONE_SURROGATE:{entity.id}")
            elif entity_type == "IP_ADDRESS" and not replacement.startswith("192.0.2."):
                violations.append(f"UNSAFE_IP_SURROGATE:{entity.id}")
            elif entity_type == "IPV6_ADDRESS" and not replacement.lower().startswith("2001:db8:"):
                violations.append(f"UNSAFE_IPV6_SURROGATE:{entity.id}")
            elif entity_type == "MAC_ADDRESS" and not replacement.upper().startswith("02:00:00:"):
                violations.append(f"UNSAFE_MAC_SURROGATE:{entity.id}")
            elif entity_type == "API_KEY" and not replacement.startswith("sk-test-"):
                violations.append(f"UNSAFE_API_KEY_SURROGATE:{entity.id}")
            elif entity_type == "BEARER_TOKEN" and not replacement.startswith("Bearer test_"):
                violations.append(f"UNSAFE_BEARER_TOKEN_SURROGATE:{entity.id}")
            elif entity_type == "ACCESS_TOKEN" and not replacement.startswith("access_test_"):
                violations.append(f"UNSAFE_ACCESS_TOKEN_SURROGATE:{entity.id}")
            elif entity_type == "PASSWORD" and not replacement.startswith("SyntheticPass-"):
                violations.append(f"UNSAFE_PASSWORD_SURROGATE:{entity.id}")
            elif entity_type == "PRIVATE_KEY" and not replacement.startswith("test-private-key-"):
                violations.append(f"UNSAFE_PRIVATE_KEY_SURROGATE:{entity.id}")
            elif entity_type == "CONNECTION_STRING" and not replacement.startswith("connection_test_"):
                violations.append(f"UNSAFE_CONNECTION_STRING_SURROGATE:{entity.id}")
            elif entity_type == "SECRET" and not replacement.startswith("secret_test_"):
                violations.append(f"UNSAFE_SECRET_SURROGATE:{entity.id}")

        # Verify that only sensitive spans changed.
        try:
            original_skeleton = _skeleton_original(original_text, analysis)
            candidate_skeleton = _skeleton_candidate(candidate, analysis, execution)
            structural_utility = SequenceMatcher(None, original_skeleton, candidate_skeleton).ratio()
            if original_skeleton != candidate_skeleton:
                violations.append("NON_SENSITIVE_TEXT_CHANGED")
        except Exception:
            structural_utility = 0.0
            violations.append("STRUCTURAL_VALIDATION_ERROR")

        constraints = {_norm(c) for c in analysis.constraints}
        numeric_consistency = self._numeric_consistency(analysis, replacements, constraints)
        temporal_consistency = self._temporal_consistency(analysis, replacements, constraints)
        relation_consistency = min(numeric_consistency, temporal_consistency)

        if constraints & _NUMERIC_CONSTRAINTS:
            if numeric_consistency < 0.999999:
                violations.append("NUMERIC_RELATION_CHANGED")

        if constraints & {"PRESERVE_DATE_INTERVAL", "PRESERVE_DATE_ORDER"}:
            if temporal_consistency < 0.999999:
                violations.append("TEMPORAL_RELATION_CHANGED")

        privacy_prefixes = (
            "EXACT_LEAK",
            "NORMALIZED_LEAK",
            "PARTIAL_LEAK",
            "UNSAFE_",
            "MISSING_SYNTHETIC_REPLACEMENT",
            "EMPTY_SYNTHETIC_REPLACEMENT",
            "FORBIDDEN_PLACEHOLDER",
        )
        privacy_fail = any(v.startswith(privacy_prefixes) for v in violations)
        privacy_risk = 1.0 if privacy_fail else 0.0
        violations = sorted(set(violations))

        return ValidationReport(
            accepted=len(violations) == 0,
            privacy_pass=not privacy_fail,
            privacy_risk=privacy_risk,
            structural_utility=structural_utility,
            numeric_consistency=numeric_consistency,
            temporal_consistency=temporal_consistency,
            relation_consistency=relation_consistency,
            violations=violations,
        )

    @staticmethod
    def _numeric_family(entity_type: str) -> str:
        normalized = _norm(entity_type)
        if normalized in _MONEY_TYPES:
            return "MONEY"
        if normalized == "PERCENTAGE":
            return "PERCENTAGE"
        if normalized == "AGE":
            return "AGE"
        if normalized == "COUNT":
            return "COUNT"
        return "NUMBER"

    def _numeric_consistency(
        self,
        analysis: AnalysisResult,
        replacements: dict[str, str],
        constraints: set[str],
    ) -> float:
        required = constraints & _NUMERIC_CONSTRAINTS
        if not required:
            return 1.0

        entity_map = {
            entity.id: entity
            for entity in analysis.entities
            if entity.sensitive and _norm(entity.type) in _NUMERIC_TYPES
        }

        eligible = 0
        passed = 0
        for group in numeric_relation_groups(analysis):
            members = sorted(
                (entity_map[entity_id] for entity_id in group if entity_id in entity_map),
                key=lambda entity: (
                    entity.start if entity.start is not None else 10**12,
                    entity.id,
                ),
            )
            for a, b in combinations(members, 2):
                eligible += 1
                if a.id not in replacements or b.id not in replacements:
                    continue
                try:
                    oa, ob = _parse_number(a.value), _parse_number(b.value)
                    ca, cb = _parse_number(replacements[a.id]), _parse_number(replacements[b.id])
                except Exception:
                    continue

                original_delta = ob - oa
                candidate_delta = cb - ca
                checks = [candidate_delta == original_delta]
                if "PRESERVE_CHANGE_DIRECTION" in required:
                    checks.append(
                        (original_delta > 0) == (candidate_delta > 0)
                        and (original_delta < 0) == (candidate_delta < 0)
                    )
                if all(checks):
                    passed += 1

        # Unrelated numeric values are intentionally independent and therefore do
        # not create pairwise invariants merely because they share the same type.
        return passed / eligible if eligible else 1.0

    def _temporal_consistency(
        self,
        analysis: AnalysisResult,
        replacements: dict[str, str],
        constraints: set[str],
    ) -> float:
        required = constraints & {"PRESERVE_DATE_INTERVAL", "PRESERVE_DATE_ORDER"}
        if not required:
            return 1.0

        entity_map = {
            entity.id: entity
            for entity in analysis.entities
            if entity.sensitive and _norm(entity.type) in _DATE_TYPES
        }

        eligible = 0
        passed = 0
        default_date = date_parser.parse("2000-01-01")

        for group in date_relation_groups(analysis):
            members = sorted(
                (entity_map[entity_id] for entity_id in group if entity_id in entity_map),
                key=lambda entity: (
                    entity.start if entity.start is not None else 10**12,
                    entity.id,
                ),
            )
            for a, b in combinations(members, 2):
                eligible += 1
                if a.id not in replacements or b.id not in replacements:
                    continue
                try:
                    oa = date_parser.parse(a.value, default=default_date)
                    ob = date_parser.parse(b.value, default=default_date)
                    ca = date_parser.parse(replacements[a.id], default=default_date)
                    cb = date_parser.parse(replacements[b.id], default=default_date)
                except Exception:
                    continue

                original_delta = (ob - oa).days
                candidate_delta = (cb - ca).days
                checks = [candidate_delta == original_delta]
                if "PRESERVE_DATE_ORDER" in required:
                    checks.append(
                        (original_delta > 0) == (candidate_delta > 0)
                        and (original_delta < 0) == (candidate_delta < 0)
                    )
                if all(checks):
                    passed += 1

        # Independent dates are syntheticized independently. Only relation-linked
        # dates carry interval/order invariants against one another.
        return passed / eligible if eligible else 1.0

