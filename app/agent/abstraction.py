from __future__ import annotations

import hashlib
import json
from collections import Counter

from app.schemas import AbstractPattern, AnalysisResult


def _norm(value: str) -> str:
    return value.strip().upper().replace(" ", "_").replace("-", "_")


class PrivacyAbstractor:
    def build(self, analysis: AnalysisResult) -> AbstractPattern:
        sensitive = [e for e in analysis.entities if e.sensitive]
        entity_types = sorted(_norm(e.type) for e in sensitive)
        semantic_roles = sorted(_norm(e.role) for e in sensitive)
        relation_types = sorted(_norm(r.relation) for r in analysis.relations)
        constraints = sorted({_norm(c) for c in analysis.constraints})
        counts = dict(sorted(Counter(entity_types).items()))

        canonical = {
            "task": _norm(analysis.task),
            "domain": _norm(analysis.domain),
            "entity_types": entity_types,
            "semantic_roles": semantic_roles,
            "relation_types": relation_types,
            "constraints": constraints,
            "entity_type_counts": counts,
        }
        fingerprint = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        return AbstractPattern(**canonical, fingerprint=fingerprint)
