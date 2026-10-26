from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from app.privacy.backstop import StructuredBackstopDetector
from app.schemas import AnalysisResult, AnalyzedEntity, ExecutionResult, SemanticRelation


def _norm(value: str) -> str:
    return value.strip().upper().replace(" ", "_").replace("-", "_")


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or a_start >= b_end)

_ATOMIC_CONTAINER_TYPES = {
    "CONNECTION_STRING",
    "PRIVATE_KEY",
    "BEARER_TOKEN",
    "ACCESS_TOKEN",
    "API_KEY",
    "EMAIL",
    "CREDIT_CARD",
    "IBAN",
    "SSN",
    "PASSPORT",
    "BANK_ACCOUNT",
    "ROUTING_NUMBER",
    "IP_ADDRESS",
    "IPV6_ADDRESS",
    "MAC_ADDRESS",
    "UUID",
    "GPS_COORDINATE",
}


def _contains(outer_start: int, outer_end: int, inner_start: int, inner_end: int) -> bool:
    return outer_start <= inner_start and inner_end <= outer_end


@dataclass
class CoverageAuditor:
    """Coverage and residual-sensitive-data auditor.

    It never authorizes release. It only augments the current analysis with newly
    detected sensitive spans. The hard safety gate remains the final authority.
    """

    detector: StructuredBackstopDetector

    def audit_original(self, text: str, analysis: AnalysisResult) -> AnalysisResult:
        """Run a second deterministic coverage pass over the original input.

        align_and_augment already includes the primary backstop. This second pass is
        intentionally idempotent and exists so future detector extensions can be added
        without changing the agent loop.
        """

        detections = self.detector.detect(text)
        return self.merge_detections(
            text=text,
            analysis=analysis,
            detections=detections,
            source="coverage",
            id_prefix="C",
        )


    def expand_repeated_occurrences(self, text: str, analysis: AnalysisResult) -> AnalysisResult:
        """Add every exact repeated occurrence of already-detected sensitive values.

        LLM analysis may identify only one occurrence of a repeated name/email/ID. A
        candidate would then leak the other occurrence and fail the hard gate. This
        pass is deterministic and uses only values already classified as sensitive; it
        never promotes new text to sensitive on its own. Repeated occurrences inherit
        the same type/role and therefore receive the same deterministic surrogate.
        """

        result = deepcopy(analysis)
        entities = [entity for entity in result.entities if entity.sensitive]
        used_spans = {
            (entity.start, entity.end)
            for entity in entities
            if entity.start is not None and entity.end is not None
        }
        existing_ids = {entity.id for entity in entities}
        next_id = 1

        for entity in list(entities):
            value = entity.value
            if not value or len(value.strip()) < 2:
                continue

            cursor = 0
            while True:
                start = text.find(value, cursor)
                if start < 0:
                    break
                end = start + len(value)
                cursor = start + max(1, len(value))

                if (start, end) in used_spans:
                    continue
                if any(_overlaps(start, end, a, b) for a, b in used_spans):
                    continue

                while f"D{next_id}" in existing_ids:
                    next_id += 1

                duplicate = entity.model_copy(deep=True)
                duplicate.id = f"D{next_id}"
                duplicate.start = start
                duplicate.end = end
                duplicate.source = "coverage"
                next_id += 1

                result.entities.append(duplicate)
                existing_ids.add(duplicate.id)
                used_spans.add((start, end))

        result.entities.sort(
            key=lambda entity: (
                entity.start if entity.start is not None else 10**12,
                entity.id,
            )
        )
        return result


    def expand_person_aliases(self, text: str, analysis: AnalysisResult) -> AnalysisResult:
        """Add first/last-name references for already detected full PERSON names.

        This is deterministic coreference coverage, not open-ended NER.  It only
        promotes exact name tokens that belong to a full PERSON entity already present
        in the analysis.  Alias entities are linked to the full-name entity so repair
        can generate a consistent synthetic identity.
        """

        result = deepcopy(analysis)
        used_spans = {
            (entity.start, entity.end)
            for entity in result.entities
            if entity.sensitive and entity.start is not None and entity.end is not None
        }
        existing_ids = {entity.id for entity in result.entities}
        relation_keys = {
            (rel.source_id, _norm(rel.relation), rel.target_id)
            for rel in result.relations
        }
        next_id = 1

        full_people = [
            entity
            for entity in result.entities
            if entity.sensitive
            and _norm(entity.type) == "PERSON"
            and len(entity.value.split()) >= 2
        ]

        for person in full_people:
            tokens = [token.strip(".,;:()[]{}\"'") for token in person.value.split()]
            aliases = []
            if tokens:
                aliases.append(tokens[0])
            if len(tokens) >= 2:
                aliases.append(tokens[-1])

            for alias in dict.fromkeys(alias for alias in aliases if len(alias) >= 3):
                import re
                for match in re.finditer(rf"(?<![A-Za-z-]){re.escape(alias)}(?![A-Za-z-])", text):
                    start, end = match.start(), match.end()
                    if person.start is not None and person.end is not None and start >= person.start and end <= person.end:
                        continue
                    if (start, end) in used_spans:
                        continue
                    if any(_overlaps(start, end, a, b) for a, b in used_spans):
                        continue

                    while f"A{next_id}" in existing_ids:
                        next_id += 1
                    alias_id = f"A{next_id}"
                    next_id += 1
                    alias_entity = AnalyzedEntity(
                        id=alias_id,
                        type="PERSON",
                        role=f"person_alias_of:{person.id}",
                        value=text[start:end],
                        sensitive=True,
                        sensitivity=person.sensitivity,
                        confidence=1.0,
                        start=start,
                        end=end,
                        source="coverage",
                    )
                    result.entities.append(alias_entity)
                    existing_ids.add(alias_id)
                    used_spans.add((start, end))
                    key = (person.id, "PERSON_ALIAS_OF", alias_id)
                    if key not in relation_keys:
                        result.relations.append(
                            SemanticRelation(
                                source_id=person.id,
                                relation="PERSON_ALIAS_OF",
                                target_id=alias_id,
                                confidence=1.0,
                            )
                        )
                        relation_keys.add(key)

        result.entities.sort(
            key=lambda entity: (
                entity.start if entity.start is not None else 10**12,
                entity.id,
            )
        )
        return result

    def find_residual_entities(
        self,
        *,
        original_text: str,
        analysis: AnalysisResult,
        execution: ExecutionResult,
    ) -> list[AnalyzedEntity]:
        """Find structured sensitive values that remain outside replacement spans.

        The detector runs on the candidate. Any detection inside a synthetic replacement
        span is ignored. Detections in unchanged candidate regions are mapped back to the
        corresponding exact original-text offsets, allowing the next attempt to rebuild
        a clean candidate from the original input.
        """

        candidate = execution.candidate_text
        detections = self.detector.detect(candidate)
        replacement_spans = [
            (record.replacement_start, record.replacement_end)
            for record in execution.replacements
        ]
        unchanged_segments = self._unchanged_segment_map(
            original_text=original_text,
            execution=execution,
        )
        protected_original = [
            (entity.start, entity.end)
            for entity in analysis.entities
            if entity.sensitive and entity.start is not None and entity.end is not None
        ]

        residual: list[AnalyzedEntity] = []
        seen_original_spans: set[tuple[int, int]] = set()

        for detected in detections:
            assert detected.start is not None and detected.end is not None

            # Safe synthetic replacements often still look like emails, tokens, etc.
            if any(
                _overlaps(detected.start, detected.end, start, end)
                for start, end in replacement_spans
            ):
                continue

            mapped = self._map_candidate_to_original(
                detected.start,
                detected.end,
                unchanged_segments,
            )
            if mapped is None:
                continue

            original_start, original_end = mapped
            if original_text[original_start:original_end] != detected.value:
                continue

            overlapping_protected = [
                (start, end)
                for start, end in protected_original
                if _overlaps(original_start, original_end, start, end)
            ]
            if overlapping_protected:
                detected_type = _norm(detected.type)
                # Ignore detections already fully protected by a known span. But if a
                # newly detected atomic structured value strictly contains the known
                # span(s), keep it: replacing only an inner password/username is not
                # enough to protect a full connection string, token, email, etc.
                if any(
                    _contains(start, end, original_start, original_end)
                    for start, end in overlapping_protected
                ):
                    continue
                if not (
                    detected_type in _ATOMIC_CONTAINER_TYPES
                    and all(
                        _contains(original_start, original_end, start, end)
                        for start, end in overlapping_protected
                    )
                ):
                    continue

            span = (original_start, original_end)
            if span in seen_original_spans:
                continue
            seen_original_spans.add(span)

            residual.append(
                AnalyzedEntity(
                    id=f"R{len(residual) + 1}",
                    type=_norm(detected.type),
                    role=detected.role,
                    value=detected.value,
                    sensitive=True,
                    sensitivity=detected.sensitivity,
                    confidence=detected.confidence,
                    start=original_start,
                    end=original_end,
                    source="residual",
                )
            )

        return residual

    def merge_detections(
        self,
        *,
        text: str,
        analysis: AnalysisResult,
        detections: list[AnalyzedEntity],
        source: str,
        id_prefix: str,
    ) -> AnalysisResult:
        result = deepcopy(analysis)
        existing_ids = {entity.id for entity in result.entities}
        id_remap: dict[str, str] = {entity.id: entity.id for entity in result.entities}
        next_id = 1

        for detected in detections:
            if detected.start is None or detected.end is None:
                continue
            start, end = detected.start, detected.end
            if not (0 <= start < end <= len(text)):
                continue
            if text[start:end] != detected.value:
                continue

            sensitive_entities = [entity for entity in result.entities if entity.sensitive]
            exact = next(
                (
                    entity
                    for entity in sensitive_entities
                    if entity.start == start and entity.end == end
                ),
                None,
            )
            if exact is not None:
                if _norm(exact.type) in {"SECRET", "UNKNOWN_SENSITIVE"}:
                    exact.type = _norm(detected.type)
                    exact.role = detected.role
                    exact.sensitivity = detected.sensitivity
                    exact.confidence = max(exact.confidence, detected.confidence)
                    exact.source = source  # type: ignore[assignment]
                continue

            overlaps = [
                entity
                for entity in sensitive_entities
                if entity.start is not None
                and entity.end is not None
                and _overlaps(start, end, entity.start, entity.end)
            ]

            if (
                overlaps
                and _norm(detected.type) in _ATOMIC_CONTAINER_TYPES
                and all(
                    entity.start is not None
                    and entity.end is not None
                    and _contains(start, end, entity.start, entity.end)
                    for entity in overlaps
                )
            ):
                keeper = min(
                    overlaps,
                    key=lambda entity: (
                        entity.start if entity.start is not None else 10**12,
                        entity.id,
                    ),
                )
                removed_ids = {entity.id for entity in overlaps}
                for removed_id in removed_ids:
                    id_remap[removed_id] = keeper.id

                result.entities = [
                    entity for entity in result.entities if entity.id not in removed_ids
                ]
                replacement = detected.model_copy(deep=True)
                replacement.id = keeper.id
                replacement.type = _norm(replacement.type)
                replacement.source = source  # type: ignore[assignment]
                result.entities.append(replacement)
                existing_ids.difference_update(removed_ids)
                existing_ids.add(replacement.id)
                continue

            if overlaps:
                continue

            while f"{id_prefix}{next_id}" in existing_ids:
                next_id += 1

            new_entity = detected.model_copy(deep=True)
            new_entity.id = f"{id_prefix}{next_id}"
            new_entity.type = _norm(new_entity.type)
            new_entity.source = source  # type: ignore[assignment]
            next_id += 1

            result.entities.append(new_entity)
            existing_ids.add(new_entity.id)
            id_remap[new_entity.id] = new_entity.id

        final_ids = {entity.id for entity in result.entities if entity.sensitive}
        remapped_relations: list[SemanticRelation] = []
        relation_keys: set[tuple[str, str, str]] = set()
        for relation in result.relations:
            source_id = id_remap.get(relation.source_id, relation.source_id)
            target_id = id_remap.get(relation.target_id, relation.target_id)
            if source_id not in final_ids or target_id not in final_ids or source_id == target_id:
                continue
            key = (source_id, _norm(relation.relation), target_id)
            if key in relation_keys:
                continue
            relation_keys.add(key)
            remapped_relations.append(
                SemanticRelation(
                    source_id=source_id,
                    relation=relation.relation,
                    target_id=target_id,
                    confidence=relation.confidence,
                )
            )
        result.relations = remapped_relations

        result.entities.sort(
            key=lambda entity: (
                entity.start if entity.start is not None else 10**12,
                entity.id,
            )
        )
        return result

    def merge_analysis(self, base: AnalysisResult, extra: AnalysisResult) -> AnalysisResult:
        """Union a coverage re-analysis into the current analysis by exact spans.

        Existing entity IDs stay stable. New entities get R-prefixed IDs. Extra
        relations are remapped to those stable IDs when both endpoints can be mapped.
        """

        result = deepcopy(base)
        existing_by_span = {
            (entity.start, entity.end): entity
            for entity in result.entities
            if entity.sensitive and entity.start is not None and entity.end is not None
        }
        used_spans = list(existing_by_span)
        existing_ids = {entity.id for entity in result.entities}
        extra_id_map: dict[str, str] = {}
        next_id = 1

        for entity in sorted(
            (item for item in extra.entities if item.sensitive),
            key=lambda item: (item.start if item.start is not None else 10**12, item.id),
        ):
            if entity.start is None or entity.end is None:
                continue
            span = (entity.start, entity.end)

            exact = existing_by_span.get(span)
            if exact is not None:
                extra_id_map[entity.id] = exact.id
                # Prefer a more specific structured type over generic categories.
                if _norm(exact.type) in {"SECRET", "UNKNOWN_SENSITIVE"}:
                    exact.type = _norm(entity.type)
                    exact.role = entity.role
                    exact.confidence = max(exact.confidence, entity.confidence)
                continue

            if any(_overlaps(entity.start, entity.end, a, b) for a, b in used_spans):
                continue

            while f"R{next_id}" in existing_ids:
                next_id += 1

            new_entity = entity.model_copy(deep=True)
            new_entity.id = f"R{next_id}"
            new_entity.source = "residual"
            next_id += 1

            result.entities.append(new_entity)
            existing_ids.add(new_entity.id)
            existing_by_span[span] = new_entity
            used_spans.append(span)
            extra_id_map[entity.id] = new_entity.id

        existing_relation_keys = {
            (relation.source_id, _norm(relation.relation), relation.target_id)
            for relation in result.relations
        }
        for relation in extra.relations:
            source_id = extra_id_map.get(relation.source_id)
            target_id = extra_id_map.get(relation.target_id)
            if not source_id or not target_id or source_id == target_id:
                continue
            key = (source_id, _norm(relation.relation), target_id)
            if key in existing_relation_keys:
                continue
            result.relations.append(
                SemanticRelation(
                    source_id=source_id,
                    relation=relation.relation,
                    target_id=target_id,
                    confidence=relation.confidence,
                )
            )
            existing_relation_keys.add(key)

        constraints = list(result.constraints)
        existing_constraints = {_norm(value) for value in constraints}
        for constraint in extra.constraints:
            if _norm(constraint) not in existing_constraints:
                constraints.append(constraint)
                existing_constraints.add(_norm(constraint))
        result.constraints = constraints

        result.entities.sort(
            key=lambda entity: (
                entity.start if entity.start is not None else 10**12,
                entity.id,
            )
        )
        return result

    @staticmethod
    def _unchanged_segment_map(
        *,
        original_text: str,
        execution: ExecutionResult,
    ) -> list[tuple[int, int, int, int]]:
        records = sorted(execution.replacements, key=lambda record: record.original_start)
        segments: list[tuple[int, int, int, int]] = []
        original_cursor = 0
        candidate_cursor = 0

        for record in records:
            original_end = record.original_start
            candidate_end = record.replacement_start
            if original_end > original_cursor and candidate_end > candidate_cursor:
                segments.append(
                    (original_cursor, original_end, candidate_cursor, candidate_end)
                )
            original_cursor = record.original_end
            candidate_cursor = record.replacement_end

        candidate_text = execution.candidate_text
        if original_cursor < len(original_text) and candidate_cursor < len(candidate_text):
            segments.append(
                (original_cursor, len(original_text), candidate_cursor, len(candidate_text))
            )

        return segments

    @staticmethod
    def _map_candidate_to_original(
        candidate_start: int,
        candidate_end: int,
        segments: list[tuple[int, int, int, int]],
    ) -> tuple[int, int] | None:
        for original_start, original_end, mapped_start, mapped_end in segments:
            if candidate_start < mapped_start or candidate_end > mapped_end:
                continue
            delta = candidate_start - mapped_start
            original_candidate_start = original_start + delta
            original_candidate_end = original_candidate_start + (candidate_end - candidate_start)
            if original_candidate_end <= original_end:
                return original_candidate_start, original_candidate_end
        return None


def classify_failure(violations: list[str]) -> str | None:
    if not violations:
        return None
    if any(
        violation.startswith(("EXACT_LEAK", "NORMALIZED_LEAK", "PARTIAL_LEAK", "RESIDUAL_SENSITIVE_DATA"))
        for violation in violations
    ):
        return "RESIDUAL_SENSITIVE_DATA"
    if any(
        violation.startswith((
            "NUMERIC_ENTITIES_NOT_GROUPED",
            "DATE_ENTITIES_NOT_GROUPED",
            "PLAN_",
            "NUMERIC_RELATION_CHANGED",
            "TEMPORAL_RELATION_CHANGED",
        ))
        for violation in violations
    ):
        return "PLAN_OR_RELATION_FAILURE"
    if any(violation.startswith("UNSAFE_") for violation in violations):
        return "UNSAFE_SYNTHETIC_SURROGATE"
    if any(violation.startswith("FORBIDDEN_PLACEHOLDER") for violation in violations):
        return "FORBIDDEN_PLACEHOLDER"
    if any(violation.startswith("NON_SENSITIVE_TEXT_CHANGED") for violation in violations):
        return "STRUCTURAL_UTILITY_FAILURE"
    return "VALIDATION_FAILURE"
