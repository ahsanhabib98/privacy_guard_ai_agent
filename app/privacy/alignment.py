from __future__ import annotations

from copy import deepcopy
import re

from app.privacy.backstop import StructuredBackstopDetector, _money_is_relation_literal
from app.schemas import AnalysisResult, AnalyzedEntity, SemanticRelation


class EntityAlignmentError(ValueError):
    pass


_AUTHORITATIVE_STRUCTURED_TYPES = {
    "PERSON",
    "ORGANIZATION",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "SSN",
    "IP_ADDRESS",
    "PRIVATE_KEY",
    "DEVICE_ID",
    "TAX_ID",
    "DRIVER_LICENSE",
    "MRN",
    "CLAIM_ID",
    "POLICY_NUMBER",
    "CUSTOMER_ID",
    "UUID",
    "GPS_COORDINATE",
    "MAC_ADDRESS",
    "IPV6_ADDRESS",
    "IBAN",
    "CREDIT_CARD",
    "BANK_ACCOUNT",
    "ROUTING_NUMBER",
    "PASSPORT",
    "EMPLOYEE_ID",
    "STUDENT_ID",
    "USERNAME",
    "MONEY",
    "DATE",
    "ISO_DATE",
    "DOB",
    "API_KEY",
    "BEARER_TOKEN",
    "ACCESS_TOKEN",
    "PASSWORD",
    "CONNECTION_STRING",
    "SECRET",
}

# If a deterministic structured detector finds one of these full atomic values,
# it is safer to transform the whole value than a smaller LLM span inside it.
# Example: replace the whole connection string, not only its password substring.
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


def _norm(value: str) -> str:
    return value.strip().upper().replace(" ", "_").replace("-", "_")


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


def _contains(outer_start: int, outer_end: int, inner_start: int, inner_end: int) -> bool:
    return outer_start <= inner_start and inner_end <= outer_end



_ORGISH_PERSON_TOKENS = {
    "llc", "inc", "corp", "corporation", "university", "college", "institute",
    "laboratory", "lab", "bank", "hospital", "analytics", "company",
}

_GENERIC_LABEL_VALUES = {
    "id", "identifier", "number", "code", "key", "record", "account",
    "device", "token", "secret", "password", "username", "email", "phone",
}


def _looks_like_person_value(value: str) -> bool:
    parts = [part for part in re.split(r"\s+", value.strip()) if part]
    if not 1 <= len(parts) <= 4:
        return False
    if any(part.lower().rstrip(".") in _ORGISH_PERSON_TOKENS for part in parts):
        return False
    return all(re.fullmatch(r"[A-Z][A-Za-z'’-]{1,30}", part) is not None for part in parts)


def _plausible_llm_entity(text: str, entity: AnalyzedEntity) -> bool:
    """Reject obvious LLM label-word/span hallucinations before they reach coverage expansion."""
    value = entity.value.strip()
    normalized_type = _norm(entity.type)
    if not value or value.lower() in _GENERIC_LABEL_VALUES:
        return False

    if normalized_type == "DEVICE_ID":
        # A device identifier must look like an identifier value, not the prose word
        # "identifier" (which previously caused replacements inside normal words).
        compact = re.sub(r"[^A-Za-z0-9]", "", value)
        return len(compact) >= 6 and any(ch.isdigit() for ch in compact)

    if normalized_type == "USERNAME":
        # USERNAME is highly ambiguous in ordinary prose (especially the English
        # verb "handle").  Accept an LLM username only when the source itself
        # provides explicit username/handle/login context, or when the value is an
        # explicit @handle.  This prevents values such as "their" in
        # "they handle their relationship" from becoming false sensitive entities.
        if value.startswith("@"):  # explicit social/account handle
            return re.fullmatch(r"@[A-Za-z0-9_][A-Za-z0-9._-]{2,63}", value) is not None

        start = entity.start
        if start is None:
            # If offsets are absent, locate an unambiguous occurrence only for the
            # purpose of checking the immediately preceding label context.
            matches = [m.start() for m in re.finditer(re.escape(value), text)]
            if len(matches) != 1:
                return False
            start = matches[0]

        before = text[max(0, start - 50):start]
        return re.search(
            r"(?i)\b(?:user\s*name|username|handle|login(?:\s+name)?|screen\s+name)\b"
            r"\s*(?::|=|\bis\b)\s*$",
            before,
        ) is not None

    if normalized_type == "PERSON":
        return _looks_like_person_value(value)

    # Relation-derived difference literals are utility metadata. Transforming them
    # independently makes a statement such as "$12,600 difference" mathematically false.
    if normalized_type in {"MONEY", "SALARY", "INCOME", "FINANCIAL_VALUE"}:
        start, end = entity.start, entity.end
        if start is not None and end is not None and 0 <= start < end <= len(text):
            if _money_is_relation_literal(text, start, end):
                return False

    return True

def _whitespace_equivalent_matches(text: str, value: str) -> list[tuple[int, int]]:
    """Find source spans that differ from the LLM value only in whitespace."""

    parts = re.split(r"\s+", value.strip())
    if not parts or any(not part for part in parts):
        return []

    pattern = re.compile(r"\s+".join(re.escape(part) for part in parts))
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def align_and_augment(
    text: str,
    analysis: AnalysisResult,
    backstop: StructuredBackstopDetector,
    *,
    skip_unalignable: bool = False,
) -> AnalysisResult:
    """Verify/repair LLM spans, then merge deterministic structured detections.

    The deterministic detector wins on exact structured spans. For atomic structured
    values such as connection strings and private keys, a larger deterministic span
    also wins over smaller LLM spans contained inside it. This prevents partial
    transformation of a compound credential.
    """

    result = deepcopy(analysis)
    used_spans: list[tuple[int, int, str]] = []
    aligned: list[AnalyzedEntity] = []

    for entity in result.entities:
        if not entity.sensitive:
            continue

        entity.type = _norm(entity.type)
        entity.role = entity.role.strip().lower().replace(" ", "_")

        if not _plausible_llm_entity(text, entity):
            continue

        start, end = entity.start, entity.end
        valid_offsets = (
            start is not None
            and end is not None
            and 0 <= start < end <= len(text)
            and text[start:end] == entity.value
        )

        if not valid_offsets:
            matches: list[int] = []
            cursor = 0
            while True:
                idx = text.find(entity.value, cursor)
                if idx < 0:
                    break
                matches.append(idx)
                cursor = idx + max(1, len(entity.value))

            start = end = None
            for idx in matches:
                candidate_end = idx + len(entity.value)
                if not any(_overlaps(idx, candidate_end, a, b) for a, b, _ in used_spans):
                    start, end = idx, candidate_end
                    break

            if start is None or end is None:
                for candidate_start, candidate_end in _whitespace_equivalent_matches(text, entity.value):
                    if not any(
                        _overlaps(candidate_start, candidate_end, a, b)
                        for a, b, _ in used_spans
                    ):
                        start, end = candidate_start, candidate_end
                        entity.value = text[start:end]
                        break

            if start is None or end is None:
                if skip_unalignable:
                    continue
                raise EntityAlignmentError(
                    f"Could not align entity {entity.id!r} value to exact source text"
                )

        duplicate = next(
            (existing_id for a, b, existing_id in used_spans if a == start and b == end),
            None,
        )
        if duplicate:
            if skip_unalignable:
                continue
            raise EntityAlignmentError(
                f"Duplicate entity span for {entity.id!r} and {duplicate!r}"
            )

        if any(_overlaps(start, end, a, b) for a, b, _ in used_spans):
            if skip_unalignable:
                continue
            raise EntityAlignmentError(
                f"Overlapping sensitive entity span for {entity.id!r}"
            )

        entity.start, entity.end = start, end
        # Always restore the exact source slice after alignment.
        entity.value = text[start:end]
        if entity.type in {"MONEY", "SALARY", "INCOME", "FINANCIAL_VALUE"} and _money_is_relation_literal(text, start, end):
            continue
        used_spans.append((start, end, entity.id))
        aligned.append(entity)

    # When a larger deterministic atomic span replaces one or more smaller LLM spans,
    # remap relations to the surviving ID so semantic structure is not silently lost.
    id_remap: dict[str, str] = {entity.id: entity.id for entity in aligned}
    existing_ids = {entity.id for entity in aligned}
    next_backstop = 1

    for detected in backstop.detect(text):
        assert detected.start is not None and detected.end is not None
        detected.type = _norm(detected.type)

        exact_match = next(
            (
                entity
                for entity in aligned
                if entity.start == detected.start and entity.end == detected.end
            ),
            None,
        )

        if exact_match is not None:
            if detected.type in _AUTHORITATIVE_STRUCTURED_TYPES:
                exact_match.type = detected.type
                exact_match.role = detected.role
                exact_match.sensitivity = detected.sensitivity
                exact_match.confidence = max(exact_match.confidence, detected.confidence)
                exact_match.source = "backstop"
                exact_match.value = text[detected.start:detected.end]
            continue

        overlaps = [
            entity
            for entity in aligned
            if entity.start is not None
            and entity.end is not None
            and _overlaps(detected.start, detected.end, entity.start, entity.end)
        ]

        # Prefer an authoritative full atomic structure over contained inner spans.
        if (
            overlaps
            and detected.type in _ATOMIC_CONTAINER_TYPES
            and all(
                entity.start is not None
                and entity.end is not None
                and _contains(detected.start, detected.end, entity.start, entity.end)
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

            aligned = [entity for entity in aligned if entity.id not in removed_ids]
            used_spans = [item for item in used_spans if item[2] not in removed_ids]

            replacement = detected.model_copy(deep=True)
            replacement.id = keeper.id
            replacement.value = text[detected.start:detected.end]
            aligned.append(replacement)
            used_spans.append((detected.start, detected.end, replacement.id))
            existing_ids.difference_update(removed_ids)
            existing_ids.add(replacement.id)
            continue

        if overlaps:
            continue

        while f"B{next_backstop}" in existing_ids:
            next_backstop += 1

        detected.id = f"B{next_backstop}"
        next_backstop += 1
        detected.value = text[detected.start:detected.end]
        aligned.append(detected)
        existing_ids.add(detected.id)
        id_remap[detected.id] = detected.id
        used_spans.append((detected.start, detected.end, detected.id))

    final_ids = {entity.id for entity in aligned}
    relation_keys: set[tuple[str, str, str]] = set()
    valid_relations: list[SemanticRelation] = []
    for relation in result.relations:
        source_id = id_remap.get(relation.source_id, relation.source_id)
        target_id = id_remap.get(relation.target_id, relation.target_id)
        if source_id not in final_ids or target_id not in final_ids or source_id == target_id:
            continue
        key = (source_id, _norm(relation.relation), target_id)
        if key in relation_keys:
            continue
        relation_keys.add(key)
        valid_relations.append(
            SemanticRelation(
                source_id=source_id,
                relation=relation.relation,
                target_id=target_id,
                confidence=relation.confidence,
            )
        )

    aligned.sort(
        key=lambda e: (
            e.start if e.start is not None else 10**12,
            e.id,
        )
    )

    result.entities = aligned
    result.relations = valid_relations
    return result
