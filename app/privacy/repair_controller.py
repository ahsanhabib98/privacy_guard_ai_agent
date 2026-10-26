from __future__ import annotations

from dataclasses import dataclass
import re

from app.schemas import (
    AnalysisResult,
    PlanAction,
    ReflectionResult,
    RetrievedExperience,
    SemanticRelation,
    TransformationPlan,
)
from app.tools.registry import DeterministicToolRegistry
from app.privacy.relation_graph import date_relation_groups, numeric_relation_groups, person_alias_groups


_MONEY_TYPES = {"MONEY", "SALARY", "INCOME", "FINANCIAL_VALUE"}
_NUMERIC_TYPES = _MONEY_TYPES | {"AGE", "COUNT", "PERCENTAGE", "NUMBER"}
_DATE_TYPES = {"DATE", "ISO_DATE", "DOB", "DATETIME"}
_SECRET_ROLE_TOKENS = {
    "api",
    "auth",
    "bearer",
    "credential",
    "key",
    "password",
    "secret",
    "token",
    "private_key",
    "connection",
}


def _norm(value: str) -> str:
    return value.strip().upper().replace(" ", "_").replace("-", "_")


def _looks_code_like(value: str) -> bool:
    stripped = value.strip()
    compact = re.sub(r"[^A-Za-z0-9]", "", stripped)
    return (
        bool(compact)
        and any(ch.isdigit() for ch in compact)
        and ("-" in stripped or "_" in stripped or len(compact) >= 12)
        and len(stripped.split()) <= 3
    )


def _sorted_entities(analysis: AnalysisResult):
    return sorted(
        (entity for entity in analysis.entities if entity.sensitive),
        key=lambda entity: (
            entity.start if entity.start is not None else 10**12,
            entity.id,
        ),
    )


@dataclass
class DeterministicRepairController:
    """Repairs analysis structure and failed plans without weakening the gate."""

    registry: DeterministicToolRegistry

    def augment_analysis(self, text: str, analysis: AnalysisResult) -> AnalysisResult:
        """Conservatively infer omitted money/date relations and constraints.

        Earlier versions only repaired relations if Qwen had already emitted the
        corresponding preservation constraint. Coverage-aware repair also infers a small
        set of high-confidence constraints from the source structure so newly discovered
        MONEY/DATE entities can preserve their relationships within the same API request.
        """

        repaired = analysis.model_copy(deep=True)
        entities = _sorted_entities(repaired)
        relation_pairs = {
            (relation.source_id, relation.target_id)
            for relation in repaired.relations
        } | {
            (relation.target_id, relation.source_id)
            for relation in repaired.relations
        }

        numeric_entities = [entity for entity in entities if _norm(entity.type) in _NUMERIC_TYPES]
        date_entities = [entity for entity in entities if _norm(entity.type) in _DATE_TYPES]

        inferred_numeric = self._infer_money_relations(
            text=text,
            entities=numeric_entities,
            relations=repaired.relations,
            relation_pairs=relation_pairs,
        )
        inferred_temporal = self._infer_date_relations(
            text=text,
            entities=date_entities,
            relations=repaired.relations,
            relation_pairs=relation_pairs,
        )

        # Exact-difference preservation is the numeric invariant.  Ratio/percent
        # constraints from older prompts are normalized to the stronger, explicit
        # additive-difference rule because both cannot generally be preserved after
        # a non-zero synthetic shift.
        legacy_numeric = {"PRESERVE_NUMERIC_RATIO", "PRESERVE_PERCENT_CHANGE"}
        had_numeric_constraint = any(
            _norm(item) in legacy_numeric | {"PRESERVE_NUMERIC_DIFFERENCE", "PRESERVE_CHANGE_DIRECTION"}
            for item in repaired.constraints
        )
        constraints = [item for item in repaired.constraints if _norm(item) not in legacy_numeric]
        normalized_constraints = {_norm(item) for item in constraints}

        if inferred_numeric or had_numeric_constraint:
            for value in ("PRESERVE_NUMERIC_DIFFERENCE", "PRESERVE_CHANGE_DIRECTION"):
                if value not in normalized_constraints:
                    constraints.append(value)
                    normalized_constraints.add(value)

        if inferred_temporal:
            for value in ("PRESERVE_DATE_INTERVAL", "PRESERVE_DATE_ORDER"):
                if value not in normalized_constraints:
                    constraints.append(value)
                    normalized_constraints.add(value)

        repaired.constraints = constraints
        return repaired

    @staticmethod
    def _infer_money_relations(
        *,
        text: str,
        entities: list,
        relations: list[SemanticRelation],
        relation_pairs: set[tuple[str, str]],
    ) -> bool:
        if len(entities) < 2:
            return False

        inferred_any = False
        for index in range(len(entities) - 1):
            left = entities[index]
            right = entities[index + 1]
            if (left.id, right.id) in relation_pairs:
                inferred_any = True
                continue
            if left.end is None or right.start is None:
                continue

            bridge = text[left.end:right.start].lower()
            # High-confidence relation: the values are close and the bridge explicitly
            # expresses a transition. This avoids pairing unrelated financial values in
            # neighboring clauses.
            compact_bridge = " ".join(bridge.split())
            transition = (
                " to " in f" {compact_bridge} "
                or compact_bridge in {"to", "->", "→"}
                or "increased to" in compact_bridge
                or "decreased to" in compact_bridge
                or "changed to" in compact_bridge
                or "rose to" in compact_bridge
                or "fell to" in compact_bridge
            )
            if not transition or len(compact_bridge) > 40:
                continue

            relations.append(
                SemanticRelation(
                    source_id=left.id,
                    relation="DETERMINISTIC_NUMERIC_RELATION",
                    target_id=right.id,
                    confidence=0.99,
                )
            )
            relation_pairs.add((left.id, right.id))
            relation_pairs.add((right.id, left.id))
            inferred_any = True

        return inferred_any

    @staticmethod
    def _infer_date_relations(
        *,
        text: str,
        entities: list,
        relations: list[SemanticRelation],
        relation_pairs: set[tuple[str, str]],
    ) -> bool:
        if len(entities) < 2:
            return False

        inferred_any = False
        for index in range(len(entities) - 1):
            left = entities[index]
            right = entities[index + 1]
            if (left.id, right.id) in relation_pairs:
                inferred_any = True
                continue
            if left.end is None or right.start is None:
                continue

            bridge = " ".join(text[left.end:right.start].lower().split())
            before = text[max(0, left.start - 25):left.start].lower() if left.start is not None else ""

            infer = False
            # Infer only high-confidence temporal relationships. Mere proximity is
            # not enough: an independent DOB, bonus date, and meeting date must not
            # become one relation group just because they occur in the same paragraph.
            explicit_interval = (
                ("between" in before and bridge.startswith("and"))
                or ("from" in before and bridge.startswith("to"))
                or bridge in {"to", "through", "until", "-", "–", "—"}
            )
            contextual_sequence = any(
                phrase in bridge
                for phrase in (
                    "scheduled for",
                    "appointment scheduled for",
                    "meeting scheduled for",
                    "returning on",
                    "ending on",
                    "deadline on",
                )
            )
            if explicit_interval or contextual_sequence:
                infer = True

            if not infer:
                continue

            relations.append(
                SemanticRelation(
                    source_id=left.id,
                    relation="DETERMINISTIC_DATE_RELATION",
                    target_id=right.id,
                    confidence=0.99,
                )
            )
            relation_pairs.add((left.id, right.id))
            relation_pairs.add((right.id, left.id))
            inferred_any = True

        return inferred_any

    def repair_plan(
        self,
        *,
        analysis: AnalysisResult,
        plan: TransformationPlan,
        violations: list[str],
        experiences: list[RetrievedExperience] | None = None,
        immediate_reflection: ReflectionResult | None = None,
    ) -> TransformationPlan:
        """Build a conservative allow-listed plan after a failed gate/audit check."""

        if not violations:
            return plan

        entities = _sorted_entities(analysis)
        constraints = {_norm(item) for item in analysis.constraints}
        actions: list[PlanAction] = []
        handled: set[str] = set()
        tool_scores = self._experience_tool_scores(experiences or [])
        if immediate_reflection is not None:
            for tool in immediate_reflection.recommended_tools:
                tool_scores[tool] = tool_scores.get(tool, 0.0) + 2.0

        # PERSON aliases/coreferences are transformed as one identity group so a full
        # name and later short references remain semantically consistent.
        for group in person_alias_groups(analysis):
            target_ids = [entity.id for entity in entities if entity.id in group]
            if len(target_ids) < 2:
                continue
            actions.append(
                PlanAction(
                    target_ids=target_ids,
                    tool="synthetic_identity",
                    rationale=(
                        "Deterministic repair groups aliases/coreferences of the same "
                        "person so the synthetic identity is consistent across mentions."
                    ),
                )
            )
            handled.update(target_ids)

        numeric_required = bool(
            constraints & {"PRESERVE_NUMERIC_DIFFERENCE", "PRESERVE_CHANGE_DIRECTION"}
        )
        temporal_required = bool(
            constraints & {"PRESERVE_DATE_INTERVAL", "PRESERVE_DATE_ORDER"}
        )

        if numeric_required:
            for group in self._numeric_groups(analysis, entities):
                if len(group) < 2:
                    continue
                target_ids = [entity.id for entity in entities if entity.id in group]
                actions.append(
                    PlanAction(
                        target_ids=target_ids,
                        tool="shared_numeric_shift",
                        rationale=(
                            "Deterministic repair applies one additive offset to the "
                            "related numeric group so every absolute difference and "
                            "ordering is preserved exactly."
                        ),
                    )
                )
                handled.update(target_ids)

        if temporal_required:
            for group in date_relation_groups(analysis):
                target_ids = [entity.id for entity in entities if entity.id in group]
                actions.append(
                    PlanAction(
                        target_ids=target_ids,
                        tool="shared_date_shift",
                        rationale=(
                            "Deterministic repair applies one shared day offset only to "
                            "dates connected by a temporal relation, preserving their "
                            "exact interval and ordering."
                        ),
                    )
                )
                handled.update(target_ids)

        for entity in entities:
            if entity.id in handled:
                continue

            tool = self._preferred_tool(entity.type, entity.role, entity.value, tool_scores)
            actions.append(
                PlanAction(
                    target_ids=[entity.id],
                    tool=tool,
                    rationale="Deterministic type-specific repair after safety-gate feedback.",
                )
            )
            handled.add(entity.id)

        feedback = ",".join(sorted(set(violations)))
        pass_count = sum(1 for item in (experiences or []) if item.result == "PASS")
        fail_count = sum(1 for item in (experiences or []) if item.result == "FAIL")
        immediate_reason = (
            immediate_reflection.outcome_reason if immediate_reflection is not None else ""
        )
        return TransformationPlan(
            strategy="deterministic_gate_repair",
            actions=actions,
            required_constraints=list(analysis.constraints),
            confidence=0.99,
            rationale=(
                f"Repaired after value-free validation feedback: {feedback}. "
                f"Used {pass_count} retrieved PASS experience(s), {fail_count} retrieved "
                f"FAIL experience(s), and the immediate prior-attempt reflection. "
                f"Immediate outcome reason: {immediate_reason}"
            ),
        )

    @staticmethod
    def _experience_tool_scores(
        experiences: list[RetrievedExperience],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for experience in experiences:
            # Successful structurally similar experiences are positive evidence;
            # failed experiences are negative evidence. Similarity weights both.
            if experience.result == "PASS":
                direction = 1.0
            elif experience.result == "FAIL":
                direction = -1.0
            else:
                # RELEASED is useful as a lesson/context record but is neutral
                # evidence: it is neither a validated PASS nor a FAIL.
                direction = 0.0
            evidence = max(1, experience.success_count + experience.failure_count)
            weight = direction * experience.similarity * (1.0 + min(evidence, 5) * 0.1)
            for tool in experience.tools:
                scores[tool] = scores.get(tool, 0.0) + weight
        return scores

    def _preferred_tool(
        self,
        entity_type: str,
        role: str,
        original_value: str,
        tool_scores: dict[str, float] | None = None,
    ) -> str:
        normalized_type = _norm(entity_type)
        allowed = self.registry.allowed_tools_for(normalized_type)

        if normalized_type in _NUMERIC_TYPES and "shared_numeric_shift" in allowed:
            return "shared_numeric_shift"
        if normalized_type in _DATE_TYPES and "shared_date_shift" in allowed:
            return "shared_date_shift"

        if len(allowed) == 1:
            return next(iter(allowed))

        role_text = role.strip().lower().replace("-", "_")

        # Natural-language sensitive attributes should remain meaningful after
        # transformation. Code-like values still use identifier/secret surrogates.
        if "synthetic_context" in allowed and not _looks_code_like(original_value):
            if normalized_type in {
                "HEALTH_INFORMATION",
                "DEMOGRAPHIC_SENSITIVE",
                "QUASI_IDENTIFIER",
                "EMPLOYMENT_INFORMATION",
                "EDUCATION_INFORMATION",
                "LEGAL_INFORMATION",
                "INSURANCE_INFORMATION",
                "BUSINESS_CONFIDENTIAL",
                "RESEARCH_CONFIDENTIAL",
            }:
                return "synthetic_context"
            if normalized_type == "UNKNOWN_SENSITIVE" and len(original_value.split()) >= 2:
                return "synthetic_context"
        if "synthetic_secret" in allowed and any(
            token in role_text for token in _SECRET_ROLE_TOKENS
        ):
            return "synthetic_secret"

        # When a type legitimately has more than one allowed tool, use privacy-safe
        # historical PASS/FAIL experience to break the tie. Never select a tool that
        # is not already allow-listed for the current entity type.
        scores = tool_scores or {}
        if len(allowed) > 1:
            ranked = sorted(
                allowed,
                key=lambda tool: (scores.get(tool, 0.0), tool == "synthetic_identifier", tool),
                reverse=True,
            )
            if scores.get(ranked[0], 0.0) != 0.0:
                return ranked[0]

        if "synthetic_identifier" in allowed:
            return "synthetic_identifier"
        if "synthetic_secret" in allowed:
            return "synthetic_secret"

        return sorted(allowed)[0]

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

    def _numeric_groups(self, analysis: AnalysisResult, entities: list) -> list[set[str]]:
        """Group only numeric entities connected by a numeric relation.

        Unrelated MONEY/NUMBER values remain independent even when a global
        preservation constraint exists elsewhere in the same input.
        """
        return numeric_relation_groups(analysis)
