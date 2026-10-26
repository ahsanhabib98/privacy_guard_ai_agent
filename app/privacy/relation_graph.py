from __future__ import annotations

from app.schemas import AnalysisResult, SemanticRelation


MONEY_TYPES = {"MONEY", "SALARY", "INCOME", "FINANCIAL_VALUE"}
NUMERIC_TYPES = MONEY_TYPES | {"AGE", "COUNT", "PERCENTAGE", "NUMBER"}
DATE_TYPES = {"DATE", "ISO_DATE", "DOB", "DATETIME"}


def norm(value: str) -> str:
    return value.strip().upper().replace(" ", "_").replace("-", "_")


def _numeric_relation_name(name: str) -> bool:
    normalized = norm(name)
    if normalized == "DETERMINISTIC_NUMERIC_RELATION":
        return True
    tokens = (
        "NUMERIC",
        "MONEY",
        "SALARY",
        "INCOME",
        "FINANCIAL",
        "INCREASE",
        "DECREASE",
        "DIFFERENCE",
        "CHANGE",
        "PERCENT",
        "RATIO",
    )
    return any(token in normalized for token in tokens)


def _date_relation_name(name: str) -> bool:
    normalized = norm(name)
    if normalized == "DETERMINISTIC_DATE_RELATION":
        return True
    tokens = (
        "DATE",
        "TEMPORAL",
        "BEFORE",
        "AFTER",
        "INTERVAL",
        "SCHEDULE",
        "START",
        "END",
        "DEADLINE",
    )
    return any(token in normalized for token in tokens)


def _relation_matches(kind: str, relation: SemanticRelation) -> bool:
    if kind == "numeric":
        return _numeric_relation_name(relation.relation)
    if kind == "date":
        return _date_relation_name(relation.relation)
    raise ValueError(f"Unsupported relation kind: {kind}")


def relation_components(
    analysis: AnalysisResult,
    *,
    eligible_types: set[str],
    kind: str,
) -> list[set[str]]:
    """Return connected components formed only by preservation relations.

    Entity type alone never creates a group. Two values are grouped only when
    the analysis contains a relevant semantic/deterministic relation between
    them (possibly through a chain of related entities).
    """

    eligible_ids = {
        entity.id
        for entity in analysis.entities
        if entity.sensitive and norm(entity.type) in eligible_types
    }
    adjacency = {entity_id: set() for entity_id in eligible_ids}

    for relation in analysis.relations:
        if (
            relation.source_id in eligible_ids
            and relation.target_id in eligible_ids
            and _relation_matches(kind, relation)
        ):
            adjacency[relation.source_id].add(relation.target_id)
            adjacency[relation.target_id].add(relation.source_id)

    components: list[set[str]] = []
    visited: set[str] = set()

    for entity_id in sorted(eligible_ids):
        if entity_id in visited or not adjacency[entity_id]:
            continue

        stack = [entity_id]
        component: set[str] = set()

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(adjacency[current] - visited)

        if len(component) >= 2:
            components.append(component)

    return components


def numeric_relation_groups(analysis: AnalysisResult) -> list[set[str]]:
    return relation_components(
        analysis,
        eligible_types=NUMERIC_TYPES,
        kind="numeric",
    )


def date_relation_groups(analysis: AnalysisResult) -> list[set[str]]:
    return relation_components(
        analysis,
        eligible_types=DATE_TYPES,
        kind="date",
    )


def person_alias_groups(analysis: AnalysisResult) -> list[set[str]]:
    """Return PERSON connected components formed only by alias/coreference relations."""
    person_ids = {
        entity.id
        for entity in analysis.entities
        if entity.sensitive and norm(entity.type) == "PERSON"
    }
    adjacency = {entity_id: set() for entity_id in person_ids}
    for relation in analysis.relations:
        relation_name = norm(relation.relation)
        if relation_name not in {"PERSON_ALIAS_OF", "SAME_PERSON", "COREFERS_TO"}:
            continue
        if relation.source_id in person_ids and relation.target_id in person_ids:
            adjacency[relation.source_id].add(relation.target_id)
            adjacency[relation.target_id].add(relation.source_id)

    components: list[set[str]] = []
    visited: set[str] = set()
    for entity_id in sorted(person_ids):
        if entity_id in visited or not adjacency[entity_id]:
            continue
        stack = [entity_id]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(adjacency[current] - visited)
        if len(component) >= 2:
            components.append(component)
    return components
