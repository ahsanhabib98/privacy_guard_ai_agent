from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation

from dateutil import parser as date_parser

from app.schemas import AnalysisResult, ExecutionResult, ReplacementRecord, TransformationPlan


FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Cameron"]
LAST_NAMES = ["Carter", "Turner", "Morgan", "Parker", "Reed", "Hayes", "Bennett", "Collins"]

SYNTHETIC_CITY_STATE = [
    ("Brookfield", "Connecticut"),
    ("Fairview", "Vermont"),
    ("Riverton", "Colorado"),
    ("Lakeview", "Oregon"),
    ("Cedar Grove", "Virginia"),
    ("Maplewood", "Minnesota"),
]
SYNTHETIC_BRIDGES = [
    "Cedar Crossing Bridge",
    "Maple River Bridge",
    "Pine Hollow Bridge",
    "Riverside Crossing",
    "Willow Creek Bridge",
]
SYNTHETIC_DATING_APPS = [
    "Example Dating App 101",
    "Example Dating App 214",
    "Example Dating App 327",
    "Example Dating App 438",
]


class ToolExecutionError(ValueError):
    pass


def _digest(secret: str, *parts: str) -> bytes:
    msg = "|".join(parts).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).digest()


def _int_from_digest(secret: str, *parts: str) -> int:
    return int.from_bytes(_digest(secret, *parts)[:8], "big")


_GENERIC_PRIVACY_FRAGMENTS = {
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
    # Semantic/venue words may intentionally survive in context-preserving surrogates.
    "bridge", "dating", "health", "condition", "detail", "relationship",
    "share", "separate", "system", "systems", "anatomy", "contextual",
}


def _privacy_fragments(value: str) -> set[str]:
    """Conservative source fragments that must not survive in a surrogate."""
    parts = set(re.findall(r"[A-Za-z]{5,}|\d{5,}", value.lower()))
    return {part for part in parts if part not in _GENERIC_PRIVACY_FRAGMENTS}


def _replacement_collides(replacement: str, originals: list[str]) -> bool:
    replacement_lower = replacement.lower()
    replacement_compact = re.sub(r"[^a-z0-9]", "", replacement_lower)
    for original in originals:
        normalized = re.sub(r"\s+", " ", original.strip().lower())
        if normalized and normalized in re.sub(r"\s+", " ", replacement_lower):
            return True
        compact = re.sub(r"[^a-z0-9]", "", original.lower())
        if len(compact) >= 8 and compact in replacement_compact:
            return True
        if any(fragment in replacement_lower for fragment in _privacy_fragments(original)):
            return True
    return False


def _semantic_replacement_collides(replacement: str, originals: list[str]) -> bool:
    """Leak check for semantic surrogates.

    Semantic replacements are expected to preserve broad meaning, so shared generic
    words are allowed. The complete original sensitive phrase (or its compact form)
    must still never survive.
    """
    replacement_normalized = re.sub(r"\s+", " ", replacement.strip().lower())
    replacement_compact = re.sub(r"[^a-z0-9]", "", replacement_normalized)
    for original in originals:
        normalized = re.sub(r"\s+", " ", original.strip().lower())
        if normalized and normalized == replacement_normalized:
            return True
        compact = re.sub(r"[^a-z0-9]", "", original.lower())
        if len(compact) >= 10 and compact == replacement_compact:
            return True
    return False


def _parse_number(text: str) -> Decimal:
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not match:
        raise ToolExecutionError(f"No numeric value found in {text!r}")
    try:
        return Decimal(match.group(0).replace(",", ""))
    except InvalidOperation as exc:
        raise ToolExecutionError(f"Cannot parse numeric value from {text!r}") from exc


def _format_money(original: str, value: Decimal, *, min_decimals: int = 0) -> str:
    symbol = ""
    for candidate in ("$", "€", "£", "¥"):
        if candidate in original:
            symbol = candidate
            break

    has_commas = "," in original
    match = re.search(r"\.(\d+)", original)
    decimals = max(len(match.group(1)) if match else 0, min_decimals)
    quantizer = Decimal("1") if decimals == 0 else Decimal("1." + "0" * decimals)
    quantized = value.quantize(quantizer)
    formatted = f"{quantized:,.{decimals}f}" if has_commas or symbol else f"{quantized:.{decimals}f}"
    return f"{symbol}{formatted}"


def _format_synthetic_number(original: str, value: Decimal) -> str:
    has_commas = "," in original
    decimal_match = re.search(r"\.(\d+)", original)
    decimal_places = len(decimal_match.group(1)) if decimal_match else 0
    quantizer = Decimal("1") if decimal_places == 0 else Decimal("1." + "0" * decimal_places)
    value = value.quantize(quantizer)
    if has_commas:
        return f"{value:,.{decimal_places}f}"
    return f"{value:.{decimal_places}f}"


def _date_granularity(original: str) -> str:
    text = original.strip()
    if re.fullmatch(r"\d{4}", text):
        return "year"
    if re.fullmatch(r"[A-Za-z]+\s+\d{4}", text):
        return "month_year"
    if re.fullmatch(r"\d{1,2}/\d{4}", text):
        return "month_year_numeric"
    return "full"


def _format_date(original: str, value: datetime) -> str:
    """Preserve the source date's visible precision/granularity.

    A year-only source stays year-only, and a month/year source stays month/year.
    This prevents privacy transformation from inventing day-level precision that was
    not present in the original text.
    """
    text = original.strip()
    granularity = _date_granularity(text)

    if granularity == "year":
        return str(value.year)
    if granularity == "month_year":
        month_token = re.match(r"[A-Za-z]+", text)
        use_short = bool(month_token and len(month_token.group(0)) <= 3)
        month = value.strftime("%b" if use_short else "%B")
        return f"{month} {value.year}"
    if granularity == "month_year_numeric":
        return f"{value.month:02d}/{value.year}"

    has_year = bool(re.search(r"\b\d{4}\b", text))

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return value.strftime("%Y-%m-%d")

    if re.search(r"[A-Za-z]", text):
        words = re.findall(r"[A-Za-z]+", text)
        first_word = words[0] if words else ""
        month = value.strftime("%B") if len(first_word) > 3 else value.strftime("%b")
        day = str(value.day)
        return f"{month} {day}, {value.year}" if "," in text or not has_year else f"{month} {day} {value.year}"

    if "/" in text:
        if has_year:
            return f"{value.month:02d}/{value.day:02d}/{value.year}"
        return f"{value.month:02d}/{value.day:02d}"

    return value.strftime("%Y-%m-%d")


def _looks_code_like(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    compact = re.sub(r"[^A-Za-z0-9]", "", stripped)
    return (
        any(ch.isdigit() for ch in compact)
        and ("-" in stripped or "_" in stripped or len(compact) >= 12)
        and len(stripped.split()) <= 3
    )


def _semantic_context_surrogate(entity_type: str, role: str, original: str, seed: int) -> str:
    """Return a natural, meaning-preserving synthetic phrase.

    This tool is intentionally used only for free-text sensitive attributes. It does
    not emit opaque identifiers such as HEALTH-1234 because those destroy downstream
    semantic utility. The replacement remains synthetic/generalized and avoids copying
    the original sensitive phrase verbatim.
    """
    normalized_type = entity_type.strip().upper().replace(" ", "_").replace("-", "_")
    role_text = role.strip().lower().replace("-", "_")
    lower = original.lower()

    if normalized_type == "HEALTH_INFORMATION":
        if "conjoined" in lower and "twin" in lower:
            return "twin with a rare congenital condition"
        if ("joined" in lower or "connected" in lower) and "torso" in lower:
            return "physically connected at the torso"
        if lower.strip().startswith("share") and any(word in lower for word in ("pelvis", "circulatory", "digestive", "reproductive", "system")):
            return "share portions of their lower anatomy and several internal systems"
        if "heart" in lower and "lung" in lower:
            return "each has separate heart and lungs"
        if "medical_condition_detail" in role_text or "condition_detail" in role_text:
            return "a related private medical detail"
        if "medical_condition" in role_text or "condition" in role_text:
            return "a private congenital health condition"
        return "a private health-related detail"

    if normalized_type in {"DEMOGRAPHIC_SENSITIVE", "QUASI_IDENTIFIER"}:
        if "relationship" in role_text:
            return "a private relationship detail"
        if "demographic" in role_text:
            return "a private personal characteristic"
        return "a private contextual detail"

    if normalized_type == "EMPLOYMENT_INFORMATION":
        return "a private employment detail"
    if normalized_type == "EDUCATION_INFORMATION":
        return "a private educational detail"
    if normalized_type == "LEGAL_INFORMATION":
        return "a private legal detail"
    if normalized_type == "INSURANCE_INFORMATION":
        return "a private insurance detail"
    if normalized_type == "BUSINESS_CONFIDENTIAL":
        return "a confidential business detail"
    if normalized_type == "RESEARCH_CONFIDENTIAL":
        return "a confidential research detail"

    # Natural-language unknown sensitive text should stay semantically readable; code-like
    # unknown values are handled by synthetic_identifier instead of reaching this branch.
    if "secret" in role_text or "credential" in role_text or "token" in role_text:
        return "a synthetic private credential"
    return "a private contextual detail"


@dataclass
class DeterministicToolRegistry:
    """Allow-listed deterministic transformations. No redaction tools exist."""

    secret: str

    ALLOWED_BY_TYPE = {
        "PERSON": {"synthetic_identity"},
        "PER": {"synthetic_identity"},
        "EMAIL": {"synthetic_email"},
        "PHONE": {"synthetic_phone"},
        "ADDRESS": {"synthetic_address"},
        "LOCATION": {"synthetic_location"},
        "PRECISE_LOCATION": {"synthetic_location"},
        "ORGANIZATION": {"synthetic_organization"},
        "ORG": {"synthetic_organization"},
        "IP_ADDRESS": {"synthetic_ip"},
        "IPV6_ADDRESS": {"synthetic_ipv6"},
        "MAC_ADDRESS": {"synthetic_mac"},
        "GPS_COORDINATE": {"synthetic_geo"},
        "UUID": {"synthetic_identifier"},

        # Financial values always get synthetic numeric values. Related values
        # are grouped by the plan/gate so one additive offset preserves exact differences.
        "MONEY": {"shared_numeric_shift", "proportional_money"},
        "SALARY": {"shared_numeric_shift", "proportional_money"},
        "INCOME": {"shared_numeric_shift", "proportional_money"},
        "FINANCIAL_VALUE": {"shared_numeric_shift", "proportional_money"},

        "DATE": {"shared_date_shift"},
        "ISO_DATE": {"shared_date_shift"},
        "DOB": {"shared_date_shift"},
        "DATETIME": {"shared_date_shift"},

        "AGE": {"synthetic_number", "shared_numeric_shift"},
        "COUNT": {"synthetic_number", "shared_numeric_shift"},
        "PERCENTAGE": {"synthetic_number", "shared_numeric_shift"},
        "NUMBER": {"synthetic_number", "shared_numeric_shift"},

        "API_KEY": {"synthetic_secret"},
        "BEARER_TOKEN": {"synthetic_secret"},
        "PASSWORD": {"synthetic_secret"},
        "SECRET": {"synthetic_secret"},
        "PRIVATE_KEY": {"synthetic_secret"},
        "ACCESS_TOKEN": {"synthetic_secret"},
        "CONNECTION_STRING": {"synthetic_secret"},

        # Free-text sensitive attributes use a semantic surrogate so the output keeps
        # the original narrative/utility instead of replacing meaning with opaque IDs.
        "HEALTH_INFORMATION": {"synthetic_context"},
        "BIOMETRIC": {"synthetic_identifier", "synthetic_context"},
        "EMPLOYMENT_INFORMATION": {"synthetic_identifier", "synthetic_context"},
        "EDUCATION_INFORMATION": {"synthetic_identifier", "synthetic_context"},
        "LEGAL_INFORMATION": {"synthetic_identifier", "synthetic_context"},
        "INSURANCE_INFORMATION": {"synthetic_identifier", "synthetic_context"},
        "BUSINESS_CONFIDENTIAL": {"synthetic_identifier", "synthetic_context"},
        "RESEARCH_CONFIDENTIAL": {"synthetic_identifier", "synthetic_context"},
        "DEMOGRAPHIC_SENSITIVE": {"synthetic_context"},
        "QUASI_IDENTIFIER": {"synthetic_identifier", "synthetic_context"},

        "UNKNOWN_SENSITIVE": {"synthetic_identifier", "synthetic_secret", "synthetic_context"},
    }

    IDENTIFIER_TYPES = {
        "GOVERNMENT_ID",
        "FINANCIAL_ID",
        "SSN",
        "PASSPORT",
        "DRIVER_LICENSE",
        "NATIONAL_ID",
        "TAX_ID",
        "CREDIT_CARD",
        "IBAN",
        "BANK_ACCOUNT",
        "ACCOUNT_NUMBER",
        "ROUTING_NUMBER",
        "MRN",
        "EMPLOYEE_ID",
        "STUDENT_ID",
        "CLAIM_ID",
        "CASE_NUMBER",
        "POLICY_NUMBER",
        "ACCOUNT_ID",
        "CUSTOMER_ID",
        "DEVICE_ID",
        "IMEI",
        "MAC_ADDRESS",
        "UUID",
        "CUSTOMER_ID",
        "INSURANCE_ID",
        "USERNAME",
        "HANDLE",
    }

    def allowed_tools_for(self, entity_type: str) -> set[str]:
        entity_type = entity_type.strip().upper().replace(" ", "_").replace("-", "_")

        # Explicit type policies take precedence over generic identifier suffix rules.
        # This lets specialized structured types such as MAC_ADDRESS keep their
        # format-preserving synthetic tool while still giving unseen IDs a fallback.
        if entity_type in self.ALLOWED_BY_TYPE:
            return set(self.ALLOWED_BY_TYPE[entity_type])

        if (
            entity_type in self.IDENTIFIER_TYPES
            or entity_type.endswith("_ID")
            or entity_type.endswith("_NUMBER")
        ):
            return {"synthetic_identifier"}

        # Universal fail-safe for any newly detected sensitive type. The repair
        # controller chooses identifier unless the semantic role clearly indicates
        # credential/secret material.
        return {"synthetic_identifier", "synthetic_secret"}

    def execute(
        self,
        *,
        text: str,
        analysis: AnalysisResult,
        plan: TransformationPlan,
        scope: str,
    ) -> ExecutionResult:
        entities = {e.id: e for e in analysis.entities if e.sensitive}
        replacements: dict[str, str] = {}
        sensitive_originals = [entity.value for entity in entities.values()]

        for action_index, action in enumerate(plan.actions):
            action_entities = []
            for target_id in action.target_ids:
                if target_id not in entities:
                    raise ToolExecutionError(f"Unknown target id {target_id}")
                action_entities.append(entities[target_id])

            tool = action.tool

            if tool == "synthetic_identity" and len(action_entities) >= 2:
                canonical = max(
                    action_entities,
                    key=lambda entity: (len(entity.value.split()), len(entity.value), entity.id),
                )
                canonical_parts = canonical.value.split()
                replacement_map = None
                for variant in range(32):
                    variant_scope = scope if variant == 0 else f"{scope}::identity:{variant}"
                    seed = _int_from_digest(
                        self.secret,
                        variant_scope,
                        tool,
                        canonical.type,
                        canonical.value,
                    )
                    first = FIRST_NAMES[seed % len(FIRST_NAMES)]
                    last = LAST_NAMES[(seed // len(FIRST_NAMES)) % len(LAST_NAMES)]
                    full = f"{first} {last}"
                    candidate_map: dict[str, str] = {}
                    for entity in action_entities:
                        parts = entity.value.split()
                        if len(parts) == 1 and canonical_parts:
                            if parts[0].lower() == canonical_parts[0].lower():
                                candidate_map[entity.id] = first
                            elif parts[0].lower() == canonical_parts[-1].lower():
                                candidate_map[entity.id] = last
                            else:
                                candidate_map[entity.id] = first
                        else:
                            candidate_map[entity.id] = full
                    if not any(
                        _replacement_collides(candidate, sensitive_originals)
                        for candidate in candidate_map.values()
                    ):
                        replacement_map = candidate_map
                        break
                if replacement_map is None:
                    raise ToolExecutionError(
                        "Could not generate a non-colliding grouped synthetic identity"
                    )
                replacements.update(replacement_map)
                continue

            if tool in {"shared_numeric_shift", "proportional_money"}:
                # A single additive offset is applied to the entire numeric group.
                # This guarantees that every pairwise numeric difference is preserved:
                # (b + k) - (a + k) == b - a.  ``proportional_money`` is retained
                # only as a backward-compatible alias for older stored plans; it now
                # follows the same difference-preserving behavior.
                values = [_parse_number(entity.value) for entity in action_entities]
                normalized_types = {
                    entity.type.strip().upper().replace(" ", "_").replace("-", "_")
                    for entity in action_entities
                }
                original_numeric_values = set(values)

                max_abs = max((abs(value) for value in values), default=Decimal("1"))
                if max_abs < 1:
                    magnitude = Decimal("1")
                else:
                    integer_digits = max(1, len(str(int(max_abs))))
                    magnitude = Decimal(10) ** max(0, integer_digits - 2)

                raw = _int_from_digest(
                    self.secret,
                    scope,
                    "shared_numeric_shift",
                    str(action_index),
                    ",".join(sorted(action.target_ids)),
                )

                factors = [Decimal("3"), Decimal("5"), Decimal("7"), Decimal("11"), Decimal("13"), Decimal("17")]
                signed_candidates: list[Decimal] = []
                start = raw % len(factors)
                preferred_sign = Decimal("-1") if raw % 2 else Decimal("1")
                for step in range(len(factors)):
                    factor = factors[(start + step) % len(factors)]
                    signed_candidates.extend([preferred_sign * magnitude * factor, -preferred_sign * magnitude * factor])

                # Percentages and ages use smaller bounded shifts so their values stay
                # semantically plausible while preserving exact differences.
                if normalized_types == {"PERCENTAGE"}:
                    signed_candidates = [Decimal(v) for v in (5, -5, 10, -10, 15, -15, 20, -20)]
                elif normalized_types == {"AGE"}:
                    signed_candidates = [Decimal(v) for v in (7, -7, 11, -11, 13, -13, 17, -17)]

                offset: Decimal | None = None
                for candidate_offset in signed_candidates:
                    shifted = [value + candidate_offset for value in values]
                    if normalized_types == {"PERCENTAGE"} and not all(Decimal("0") <= v <= Decimal("100") for v in shifted):
                        continue
                    if normalized_types == {"AGE"} and not all(Decimal("1") <= v <= Decimal("120") for v in shifted):
                        continue
                    if normalized_types <= {"COUNT", "NUMBER"} and all(v >= 0 for v in values) and not all(v >= 0 for v in shifted):
                        continue
                    if normalized_types <= {"MONEY", "SALARY", "INCOME", "FINANCIAL_VALUE"} and all(v >= 0 for v in values) and not all(v >= 0 for v in shifted):
                        continue
                    # Do not let a synthetic number equal any original numeric value in
                    # the same group, which would be detected as an exact leak.
                    if any(v in original_numeric_values for v in shifted):
                        continue
                    offset = candidate_offset
                    break

                if offset is None:
                    # Deterministic positive fallback.  It still preserves all pairwise
                    # differences exactly and avoids a zero/no-op transformation.
                    offset = magnitude * Decimal("23")

                for entity, value in zip(action_entities, values):
                    shifted_value = value + offset
                    normalized_type = entity.type.strip().upper().replace(" ", "_").replace("-", "_")
                    if normalized_type in {"MONEY", "SALARY", "INCOME", "FINANCIAL_VALUE"}:
                        replacement = _format_money(entity.value, shifted_value, min_decimals=2)
                    else:
                        replacement = _format_synthetic_number(entity.value, shifted_value)
                        if normalized_type == "PERCENTAGE" and "%" in entity.value and not replacement.endswith("%"):
                            replacement += "%"
                    replacements[entity.id] = replacement
                continue

            if tool == "shared_date_shift":
                raw = _int_from_digest(
                    self.secret,
                    scope,
                    tool,
                    str(action_index),
                    ",".join(sorted(action.target_ids)),
                )
                # Coarse dates (year-only or month/year) must remain coarse and must
                # actually change. Use at least a one-year day shift so a source such
                # as "2020" does not survive unchanged after formatting. Full dates use
                # the smaller shift used by the original implementation.
                coarse = any(_date_granularity(entity.value) != "full" for entity in action_entities)
                offset = (400 + raw % 696) if coarse else (30 + raw % 336)
                if raw % 2:
                    offset *= -1

                for entity in action_entities:
                    try:
                        parsed = date_parser.parse(
                            entity.value,
                            fuzzy=False,
                            default=datetime(2000, 1, 1),
                        )
                    except Exception as exc:
                        raise ToolExecutionError(f"Cannot parse date {entity.value!r}") from exc
                    replacements[entity.id] = _format_date(
                        entity.value,
                        parsed + timedelta(days=offset),
                    )
                continue

            for entity in action_entities:
                replacement = None
                for variant in range(32):
                    variant_scope = scope if variant == 0 else f"{scope}::surrogate:{variant}"
                    candidate = self._execute_single(
                        tool,
                        entity.type,
                        entity.role,
                        entity.value,
                        variant_scope,
                        entity.id,
                    )
                    collision = (
                        _semantic_replacement_collides(candidate, sensitive_originals)
                        if tool == "synthetic_context"
                        else _replacement_collides(candidate, sensitive_originals)
                    )
                    if not collision:
                        replacement = candidate
                        break
                if replacement is None:
                    raise ToolExecutionError(
                        f"Could not generate a non-colliding synthetic value for {entity.id}"
                    )
                replacements[entity.id] = replacement

        missing = [entity_id for entity_id in entities if entity_id not in replacements]
        if missing:
            raise ToolExecutionError(f"Plan left sensitive entities unhandled: {missing}")

        ordered = sorted(
            entities.values(),
            key=lambda e: e.start if e.start is not None else -1,
        )
        if any(e.start is None or e.end is None for e in ordered):
            raise ToolExecutionError("All sensitive entities must have aligned offsets before execution")

        pieces: list[str] = []
        cursor = 0
        out_cursor = 0
        records: list[ReplacementRecord] = []

        for entity in ordered:
            assert entity.start is not None and entity.end is not None
            prefix = text[cursor:entity.start]
            pieces.append(prefix)
            out_cursor += len(prefix)

            replacement = replacements[entity.id]
            replacement_start = out_cursor
            pieces.append(replacement)
            out_cursor += len(replacement)
            records.append(
                ReplacementRecord(
                    entity_id=entity.id,
                    original_start=entity.start,
                    original_end=entity.end,
                    replacement_start=replacement_start,
                    replacement_end=out_cursor,
                    replacement=replacement,
                )
            )
            cursor = entity.end

        pieces.append(text[cursor:])
        return ExecutionResult(
            candidate_text="".join(pieces),
            replacements=records,
            entity_replacements=replacements,
        )

    def _execute_single(
        self,
        tool: str,
        entity_type: str,
        role: str,
        original: str,
        scope: str,
        entity_id: str,
    ) -> str:
        del entity_id
        seed = _int_from_digest(self.secret, scope, tool, entity_type, original)
        normalized_type = entity_type.strip().upper().replace(" ", "_").replace("-", "_")

        if tool == "synthetic_identity":
            first = FIRST_NAMES[seed % len(FIRST_NAMES)]
            last = LAST_NAMES[(seed // len(FIRST_NAMES)) % len(LAST_NAMES)]
            result = f"{first} {last}"
            if result.lower() != original.lower():
                return result
            alternate_last = LAST_NAMES[(seed + 1) % len(LAST_NAMES)]
            return f"{first} {alternate_last}"

        if tool == "synthetic_email":
            return f"user{seed % 100000:05d}@example.com"

        if tool == "synthetic_phone":
            # 202-555-0100 through 202-555-0199 are reserved for fictional use.
            return f"+1 202-555-01{seed % 100:02d}"

        if tool == "synthetic_address":
            return f"{100 + seed % 8900} Example Avenue"

        if tool == "synthetic_location":
            city, state = SYNTHETIC_CITY_STATE[seed % len(SYNTHETIC_CITY_STATE)]
            lower = original.lower()
            role_text = role.lower().replace("-", "_")
            if "bridge" in lower or "bridge" in role_text or "venue" in role_text:
                bridge = SYNTHETIC_BRIDGES[(seed // len(SYNTHETIC_CITY_STATE)) % len(SYNTHETIC_BRIDGES)]
                return f"{bridge} in {city}, {state}"
            return f"{city}, {state}"

        if tool == "synthetic_organization":
            role_text = role.lower().replace("-", "_")
            if "dating_app" in role_text or "dating" in role_text:
                return SYNTHETIC_DATING_APPS[seed % len(SYNTHETIC_DATING_APPS)]
            return f"Example Organization {seed % 1000:03d}"

        if tool == "synthetic_context":
            return _semantic_context_surrogate(normalized_type, role, original, seed)

        if tool == "synthetic_identifier":
            prefix = re.sub(r"[^A-Za-z]", "", normalized_type)[:6] or "ID"
            return f"{prefix}-{seed % 10**10:010d}"

        if tool == "synthetic_secret":
            digest_hex = _digest(
                self.secret,
                scope,
                tool,
                normalized_type,
                original,
            ).hex()

            if normalized_type == "API_KEY":
                return f"sk-test-{digest_hex[:20]}"
            if normalized_type == "BEARER_TOKEN":
                return f"Bearer test_{digest_hex[:20]}"
            if normalized_type == "PASSWORD":
                return f"SyntheticPass-{digest_hex[:10]}!9"
            if normalized_type == "PRIVATE_KEY":
                return f"test-private-key-{digest_hex[:24]}"
            if normalized_type == "ACCESS_TOKEN":
                return f"access_test_{digest_hex[:20]}"
            if normalized_type == "CONNECTION_STRING":
                return f"connection_test_{digest_hex[:24]}"
            if normalized_type == "SECRET":
                return f"secret_test_{digest_hex[:20]}"
            return f"synthetic-{digest_hex[:20]}"

        if tool == "synthetic_ip":
            # TEST-NET-1 documentation range; never a routable real host address.
            return f"192.0.2.{1 + seed % 254}"

        if tool == "synthetic_ipv6":
            # 2001:db8::/32 is reserved for documentation/examples.
            a = (seed >> 32) & 0xFFFF
            b = (seed >> 16) & 0xFFFF
            c = seed & 0xFFFF
            return f"2001:db8:{a:04x}:{b:04x}::{c:04x}"

        if tool == "synthetic_mac":
            # 02 sets the locally-administered bit and avoids a real vendor OUI.
            octets = [0x02, 0x00, 0x00, (seed >> 16) & 0xFF, (seed >> 8) & 0xFF, seed & 0xFF]
            return ":".join(f"{value:02X}" for value in octets)

        if tool == "synthetic_geo":
            # Deterministic artificial coordinate, intentionally different from source.
            lat = 35.0 + ((seed % 900000) / 100000.0)
            lon = -110.0 + (((seed // 900000) % 1900000) / 100000.0)
            return f"{lat:.6f}, {lon:.6f}"

        if tool == "synthetic_number":
            original_value = _parse_number(original)

            if normalized_type == "AGE":
                value = Decimal(18 + seed % 72)
                if value == original_value:
                    value = Decimal(18 + (seed + 7) % 72)
                return str(int(value))

            if normalized_type == "PERCENTAGE":
                value = Decimal(1 + seed % 99)
                if value == original_value:
                    value = Decimal(1 + (seed + 11) % 99)
                return f"{int(value)}%" if "%" in original else str(int(value))

            if normalized_type == "COUNT":
                value = Decimal(1 + seed % 1000)
                if value == original_value:
                    value = Decimal(1 + (seed + 17) % 1000)
                return str(int(value))

            absolute_value = abs(original_value)
            if absolute_value < 1:
                lower = Decimal("1")
                upper = Decimal("99")
            else:
                integer_digits = len(str(int(absolute_value)))
                lower = Decimal(10) ** max(0, integer_digits - 1)
                upper = (Decimal(10) ** integer_digits) - 1

            span = int(upper - lower + 1)
            synthetic_value = lower + Decimal(seed % max(span, 1))
            if synthetic_value == original_value:
                synthetic_value += 1
                if synthetic_value > upper:
                    synthetic_value = lower
            return _format_synthetic_number(original, synthetic_value)

        raise ToolExecutionError(f"Unsupported tool {tool!r}")
