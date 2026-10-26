from __future__ import annotations

from app.llm.client import QwenOllamaClient
from app.llm.prompts import ANALYZE_SYSTEM, PLAN_SYSTEM, REFLECT_SYSTEM
from app.schemas import (
    AbstractPattern,
    AnalysisResult,
    ReflectionResult,
    RetrievedExperience,
    TransformationPlan,
    ValidationReport,
)


TOOL_CATALOG = {
    "synthetic_identity": "Replace a person identity with a deterministic synthetic identity.",
    "synthetic_email": "Replace an email with a deterministic example.com synthetic address.",
    "synthetic_phone": "Replace a phone number with a deterministic fictional 202-555-01xx number.",
    "synthetic_address": "Replace a street/postal address with a deterministic synthetic address.",
    "synthetic_location": "Replace a sensitive place/venue with a deterministic synthetic place of the same broad kind.",
    "synthetic_organization": "Replace an organization with a deterministic synthetic organization while preserving its semantic role (for example, dating app).",
    "synthetic_context": "Replace a free-text sensitive attribute with a natural semantic surrogate that preserves the original narrative role instead of an opaque ID.",
    "synthetic_identifier": "Replace an identifier or code-like sensitive value with a deterministic non-real synthetic surrogate.",
    "synthetic_secret": "Replace a credential or secret with a deterministic, non-functional synthetic credential-like value.",
    "synthetic_ip": "Replace an IP address with a documentation-range synthetic IP address.",
    "synthetic_ipv6": "Replace an IPv6 address with a documentation-range 2001:db8::/32 synthetic IPv6 address.",
    "synthetic_mac": "Replace a MAC address with a deterministic locally-administered synthetic MAC address.",
    "synthetic_geo": "Replace a precise coordinate with a deterministic artificial coordinate.",
    "shared_numeric_shift": "Transform related numeric values using one deterministic additive offset so every absolute difference and ordering is preserved exactly.",
    "proportional_money": "Backward-compatible alias for shared_numeric_shift; it now preserves exact numeric differences with one shared additive offset.",
    "shared_date_shift": "Shift one or more dates using one deterministic offset; related dates must be grouped to preserve intervals/order.",
    "synthetic_number": "Replace a standalone sensitive number with a deterministic synthetic numeric value, not a generalized range.",
}


class LocalQwenReasoner:
    """One local Qwen model performs ANALYZE, PLAN, and REFLECT roles."""

    def __init__(self, client: QwenOllamaClient):
        self.client = client

    async def analyze(self, text: str, repair_hint: str | None = None) -> AnalysisResult:
        payload = {"input_text": text}
        if repair_hint:
            payload["repair_hint"] = repair_hint
        return await self.client.structured_chat(
            system_prompt=ANALYZE_SYSTEM,
            user_payload=payload,
            output_model=AnalysisResult,
            schema_name="privacy_analysis",
        )

    async def plan(
        self,
        *,
        analysis: AnalysisResult,
        pattern: AbstractPattern,
        experiences: list[RetrievedExperience],
        repair_feedback: list[str] | None = None,
    ) -> TransformationPlan:
        # Intentionally omit raw entity values. Planning sees only abstract metadata.
        entity_metadata = [
            {
                "id": e.id,
                "type": e.type,
                "role": e.role,
                "sensitivity": e.sensitivity,
                "confidence": e.confidence,
                "source": e.source,
            }
            for e in analysis.entities
            if e.sensitive
        ]
        relation_metadata = [r.model_dump() for r in analysis.relations]
        payload = {
            "entity_metadata": entity_metadata,
            "relations": relation_metadata,
            "constraints": list(analysis.constraints),
            "abstract_pattern": pattern.model_dump(),
            "prior_experiences": [e.model_dump() for e in experiences],
            "repair_feedback": list(repair_feedback or []),
            "tool_catalog": TOOL_CATALOG,
        }
        return await self.client.structured_chat(
            system_prompt=PLAN_SYSTEM,
            user_payload=payload,
            output_model=TransformationPlan,
            schema_name="transformation_plan",
        )

    async def reflect(
        self,
        *,
        pattern: AbstractPattern,
        plan: TransformationPlan,
        validation: ValidationReport,
    ) -> ReflectionResult:
        payload = {
            "abstract_pattern": pattern.model_dump(),
            "plan": plan.model_dump(),
            "validation": validation.model_dump(),
        }
        return await self.client.structured_chat(
            system_prompt=REFLECT_SYSTEM,
            user_payload=payload,
            output_model=ReflectionResult,
            schema_name="reflection",
        )
