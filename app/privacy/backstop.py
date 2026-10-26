from __future__ import annotations

import re
import ipaddress
from dataclasses import dataclass

from app.schemas import AnalyzedEntity


@dataclass(frozen=True)
class _PatternSpec:
    entity_type: str
    pattern: re.Pattern[str]
    role: str
    value_group: str | None = None


# Month/date fragments are kept explicit so the detector remains deterministic.
_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_TEXT_DATE = rf"{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,)?\s+\d{{4}}"

# High-confidence lexical patterns used as a semantic backstop for names and
# domain-specific identifiers that a general LLM may occasionally miss.
_PERSON_CANDIDATE = re.compile(
    r"\b(?P<value>[A-Z][A-Za-z'’-]{1,30}\s+[A-Z][A-Za-z'’-]{1,30})\b"
)
_PERSON_CONTEXT_BEFORE = re.compile(
    r"(?i)(?:contacted|contact|copy|cc|manager\s+is|emergency\s+contact\s+is|"
    r"send(?:\s+the\s+[^.]{0,60})?\s+to|reviewed\s+by|submitted\s+by)\s*$"
)
_PERSON_CONTEXT_AFTER = re.compile(
    r"(?i)^\s+(?:lives|works|submitted|reviewed|contacted|can\s+be\s+reached|"
    r"is\s+(?:a|an|the)|was\s+(?:a|an|the))\b"
)
_ORG_NAME_TOKENS = {
    "analytics", "university", "college", "institute", "laboratory", "lab",
    "llc", "inc", "corp", "corporation", "company", "bank", "hospital",
}

_STRONG_DOMAIN_CODE = re.compile(
    r"(?<![A-Za-z0-9])(?P<value>[A-Z][A-Z0-9]{1,20}(?:[-_:][A-Z0-9]{2,32}){1,8})(?![A-Za-z0-9])"
)
_CODE_CONTEXT = re.compile(
    r"(?i)(?:identifier|\bid\b|code|key|reference|number|record|channel|asset|sample|"
    r"case|tracking|assignment|authentication|string|secret|private|confidential|restricted|"
    r"internal|proprietary|participant|template|telemetry|job)"
)


def _money_is_relation_literal(text: str, start: int, end: int) -> bool:
    """True when a money literal states a derived difference rather than a base value.

    Exact-difference preservation mathematically requires such a literal to stay equal
    to the original difference. Treat it as utility metadata rather than a separately
    transformable financial value.
    """
    before = " ".join(text[max(0, start - 35):start].lower().split())
    after = " ".join(text[end:min(len(text), end + 35)].lower().split())
    return (
        re.search(r"(?:exact|absolute|salary|numeric)?\s*difference(?:\s+of)?\s*$", before) is not None
        or re.match(r"^\s*(?:absolute\s+|exact\s+)?difference\b", after) is not None
    )


class StructuredBackstopDetector:
    """Deterministic high-confidence sensitive-data detector.

    This detector is deliberately conservative: it focuses on structured values and
    label/value patterns that can be recognized reliably without asking the LLM. It is
    used both before transformation (coverage backstop) and after transformation
    (residual-data audit).
    """

    PATTERNS: list[_PatternSpec] = [
        # Context-labeled person names. The value group intentionally requires title-case
        # tokens so ordinary text following "employee" is not broadly captured.
        _PatternSpec(
            "PERSON",
            re.compile(
                r"\b(?i:employee|contact|patient|student|name)\s+(?:is\s+)?"
                r"(?!(?i:id|identifier|number)\b)"
                r"(?P<value>[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,2})\b"
            ),
            "person_identity",
            "value",
        ),
        # Date of birth before general date so an exact DOB span keeps the more specific type.
        _PatternSpec(
            "DOB",
            re.compile(
                rf"\b(?i:born\s+on|date\s+of\s+birth|dob)\s*(?::|=|\bis\b)?\s*"
                rf"(?P<value>{_TEXT_DATE}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{2,4}})"
            ),
            "date_of_birth",
            "value",
        ),
        _PatternSpec(
            "EMAIL",
            re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w-])"),
            "contact_email",
        ),
        _PatternSpec(
            "PHONE",
            re.compile(r"(?<!\d)(?:\+?1[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]\d{4}(?!\d)"),
            "contact_phone",
        ),
        _PatternSpec(
            "ORGANIZATION",
            re.compile(
                r"\b(?:[A-Z][A-Za-z0-9&.'’-]*\s+){1,5}"
                r"(?:LLC|Inc\.?|Corporation|Corp\.?|University|College|Institute|"
                r"Laboratory|Lab|Bank|Hospital)\b"
            ),
            "organization",
        ),
        _PatternSpec(
            "ADDRESS",
            re.compile(
                r"\b\d{1,6}\s+(?:[A-Za-z0-9.'’-]+\s+){1,6}"
                r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Drive|Dr\.?|Lane|Ln\.?|"
                r"Boulevard|Blvd\.?|Court|Ct\.?|Way|Parkway|Pkwy\.?)"
                r"(?:,\s*[A-Za-z .'-]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?\b",
                re.IGNORECASE,
            ),
            "physical_address",
        ),
        _PatternSpec(
            "EMPLOYEE_ID",
            re.compile(
                r"(?i)\bemployee\s*(?:(?:identification\s+)?(?:identifier|number|id))\b\s*(?::|=|\bis\b)?\s*"
                r"(?P<value>[A-Za-z0-9][A-Za-z0-9_-]{3,})"
            ),
            "employee_identifier",
            "value",
        ),
        _PatternSpec(
            "STUDENT_ID",
            re.compile(
                r"(?i)\bstudent\s*(?:(?:identification\s+)?(?:identifier|number|id))\b\s*(?::|=|\bis\b)?\s*"
                r"(?P<value>[A-Za-z0-9][A-Za-z0-9_-]{3,})"
            ),
            "student_identifier",
            "value",
        ),
        _PatternSpec(
            "PASSPORT",
            re.compile(
                r"(?i)\bpassport(?:\s+(?:identifier|number|no\.?|id)\b)?\s*(?::|=|\bis\b)?\s*"
                r"(?P<value>[A-Z0-9][A-Z0-9-]{5,19})\b"
            ),
            "government_identifier",
            "value",
        ),
        _PatternSpec(
            "BANK_ACCOUNT",
            re.compile(
                r"(?i)\b(?:bank\s+)?account\s*(?:number|no\.?|id)?\s*(?::|=|\bis\b)?\s*"
                r"(?P<value>\d{6,20})\b"
            ),
            "financial_identifier",
            "value",
        ),
        _PatternSpec(
            "ROUTING_NUMBER",
            re.compile(
                r"(?i)\brouting\s*(?:number|no\.?)?\s*(?::|=|\bis\b)?\s*"
                r"(?P<value>\d{9})\b"
            ),
            "financial_identifier",
            "value",
        ),
        # Username labels must use an explicit delimiter.  In particular, a bare
        # English verb such as "they handle their relationship" must never be
        # interpreted as a social-media handle followed by username "their".
        _PatternSpec(
            "USERNAME",
            re.compile(
                r"(?i)\b(?:user\s*name|username|handle|login(?:\s+name)?|screen\s+name)\b"
                r"\s*(?::|=|\bis\b)\s*"
                r"(?P<value>@?[A-Za-z0-9][A-Za-z0-9._-]{2,63})\b"
            ),
            "online_identifier",
            "value",
        ),
        # Explicit @handles are also usernames, but the negative lookbehind keeps
        # this from matching the @domain portion of an email address.
        _PatternSpec(
            "USERNAME",
            re.compile(
                r"(?<![A-Za-z0-9._%+-])(?P<value>@[A-Za-z0-9_][A-Za-z0-9._-]{2,63})\b"
            ),
            "online_identifier",
            "value",
        ),
        _PatternSpec(
            "CUSTOMER_ID",
            re.compile(r"(?i)\b(?:customer|client)\s*(?:id|identifier|number|no\.?)\s*(?::|=|\bis\b)?\s*(?P<value>[A-Za-z0-9][A-Za-z0-9._-]{3,64})"),
            "customer_identifier",
            "value",
        ),
        _PatternSpec(
            "POLICY_NUMBER",
            re.compile(r"(?i)\bpolicy\s*(?:id|identifier|number|no\.?)\s*(?::|=|\bis\b)?\s*(?P<value>[A-Za-z0-9][A-Za-z0-9._-]{3,64})"),
            "insurance_identifier",
            "value",
        ),
        _PatternSpec(
            "CLAIM_ID",
            re.compile(r"(?i)\bclaim\s*(?:id|identifier|number|no\.?)\s*(?::|=|\bis\b)?\s*(?P<value>[A-Za-z0-9][A-Za-z0-9._-]{3,64})"),
            "insurance_identifier",
            "value",
        ),
        _PatternSpec(
            "MRN",
            re.compile(r"(?i)\b(?:mrn|medical\s+record\s+(?:id|number))\s*(?::|=|\bis\b)?\s*(?P<value>[A-Za-z0-9][A-Za-z0-9._-]{3,64})"),
            "medical_identifier",
            "value",
        ),
        _PatternSpec(
            "DRIVER_LICENSE",
            re.compile(r"(?i)\b(?:driver'?s?\s+license|driving\s+license)\s*(?:id|identifier|number|no\.?)?\s*(?::|=|\bis\b)?\s*(?P<value>[A-Za-z0-9][A-Za-z0-9._-]{4,32})"),
            "government_identifier",
            "value",
        ),
        _PatternSpec(
            "TAX_ID",
            re.compile(r"(?i)\b(?:tax\s+id|tin|ein)\s*(?::|=|\bis\b)?\s*(?P<value>[A-Za-z0-9][A-Za-z0-9._-]{4,32})"),
            "government_identifier",
            "value",
        ),
        _PatternSpec(
            "SSN",
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "government_identifier",
        ),
        _PatternSpec(
            "IP_ADDRESS",
            re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            "network_identifier",
        ),
        _PatternSpec(
            "MAC_ADDRESS",
            re.compile(r"(?i)\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b"),
            "device_identifier",
        ),
        _PatternSpec(
            "IPV6_ADDRESS",
            re.compile(r"(?i)(?<![0-9a-f:])[0-9a-f]{0,4}(?::[0-9a-f]{0,4}){2,7}(?![0-9a-f:])"),
            "network_identifier",
        ),
        _PatternSpec(
            "DEVICE_ID",
            re.compile(
                r"(?i)\b(?:device[_ -]?(?:id|identifier)\b|imei\b|advertising[_ -]?id\b)\s*(?::|=|\bis\b)?\s*"
                r"(?P<value>[A-Za-z0-9](?:[A-Za-z0-9._:-]{4,62}[A-Za-z0-9])?)"
            ),
            "device_identifier",
            "value",
        ),
        _PatternSpec(
            "UUID",
            re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"),
            "identifier",
        ),
        _PatternSpec(
            "GPS_COORDINATE",
            re.compile(r"(?<![\d.])-?(?:90(?:\.0+)?|[0-8]?\d(?:\.\d+)?)\s*,\s*-?(?:180(?:\.0+)?|1[0-7]\d(?:\.\d+)?|\d?\d(?:\.\d+)?)(?!\d)"),
            "precise_location",
        ),
        _PatternSpec(
            "IBAN",
            re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
            "financial_identifier",
        ),
        # Connection strings are checked before credential fragments because one full
        # connection string should be replaced atomically.
        _PatternSpec(
            "CONNECTION_STRING",
            re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s,;]+(?<![.,)])"),
            "credential_connection_string",
        ),
        _PatternSpec(
            "PRIVATE_KEY",
            re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z0-9 ]*PRIVATE KEY-----"),
            "credential",
        ),
        _PatternSpec(
            "ACCESS_TOKEN",
            re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
            "credential",
        ),
        _PatternSpec(
            "API_KEY",
            re.compile(r"\b(?:sk|pk|ghp|github_pat|xox[baprs])-[_A-Za-z0-9-]{12,}\b", re.IGNORECASE),
            "credential",
        ),
        _PatternSpec(
            "API_KEY",
            re.compile(
                r"(?i)\b(?:api[_ -]?key|apikey)\s*(?::|=|\bis\b)?\s*"
                r"(?P<value>[^\s,;]{8,})"
            ),
            "credential",
            "value",
        ),
        _PatternSpec(
            "BEARER_TOKEN",
            re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{12,}=*"),
            "credential",
        ),
        _PatternSpec(
            "ACCESS_TOKEN",
            re.compile(
                r"(?i)\b(?:access[_ -]?token|auth[_ -]?token|token)\s*(?::|=|\bis\b)?\s*"
                r"(?P<value>[^\s,;]{8,})"
            ),
            "credential",
            "value",
        ),
        _PatternSpec(
            "PASSWORD",
            re.compile(
                r"(?i)\b(?:password|passwd|pwd)\s*(?::|=|\bis\b)?\s*"
                r"(?P<value>[^\s,;]{4,})"
            ),
            "credential",
            "value",
        ),
        _PatternSpec(
            "SECRET",
            re.compile(
                r"(?i)\b(?:secret|secret\s+value)\s*(?::|=|\bis\b)\s*"
                r"(?P<value>[^\s,;]{6,})"
            ),
            "credential",
            "value",
        ),
        _PatternSpec(
            "CREDIT_CARD",
            re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
            "financial_identifier",
        ),
        # Monetary amounts are transformed because the project treats financial values
        # as sensitive. Currency symbols make this detector relatively high precision.
        _PatternSpec(
            "MONEY",
            re.compile(r"(?<!\w)[$€£¥]\s?\d[\d,]*(?:\.\d{1,2})?\b"),
            "financial_amount",
        ),
        _PatternSpec(
            "DATE",
            re.compile(rf"\b{_TEXT_DATE}\b", re.IGNORECASE),
            "calendar_date",
        ),
        _PatternSpec(
            "ISO_DATE",
            re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
            "calendar_date",
        ),
    ]

    def detect(self, text: str) -> list[AnalyzedEntity]:
        candidates: list[tuple[int, int, str, str, str]] = []

        for spec in self.PATTERNS:
            for match in spec.pattern.finditer(text):
                if spec.value_group:
                    value = match.group(spec.value_group)
                    start = match.start(spec.value_group)
                    end = match.end(spec.value_group)
                else:
                    value = match.group(0)
                    start = match.start()
                    end = match.end()

                if spec.entity_type == "CREDIT_CARD":
                    digits = re.sub(r"\D", "", value)
                    if not 13 <= len(digits) <= 19:
                        continue

                if spec.entity_type == "IP_ADDRESS":
                    parts = value.split(".")
                    if any(int(part) > 255 for part in parts):
                        continue

                if spec.entity_type == "IPV6_ADDRESS":
                    try:
                        parsed_ip = ipaddress.ip_address(value)
                    except ValueError:
                        continue
                    if parsed_ip.version != 6:
                        continue

                if spec.entity_type == "MONEY" and _money_is_relation_literal(text, start, end):
                    continue

                candidates.append((start, end, value, spec.entity_type, spec.role))

        # Person-name coverage for natural prose. Require contextual evidence so
        # title-cased organization names are not blindly promoted to PERSON.
        for match in _PERSON_CANDIDATE.finditer(text):
            value = match.group("value")
            tokens = {token.lower().strip(".'’-\"") for token in value.split()}
            if tokens & _ORG_NAME_TOKENS:
                continue
            before = text[max(0, match.start() - 90):match.start()]
            after = text[match.end():min(len(text), match.end() + 60)]
            has_email_after = re.match(
                r"(?i)^\s+at\s+[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                after,
            ) is not None
            sentence_start = match.start() == 0 or text[max(0, match.start()-2):match.start()].endswith((". ", "\n"))
            if not (
                _PERSON_CONTEXT_BEFORE.search(before)
                or _PERSON_CONTEXT_AFTER.search(after)
                or has_email_after
                or (sentence_start and re.match(r"(?i)^\s+(?:lives|works)\b", after))
            ):
                continue
            candidates.append((match.start(), match.end(), value, "PERSON", "person_identity"))

        # Unknown/domain-specific structured codes. This detector intentionally
        # requires both a strong code shape and nearby sensitive-code vocabulary.
        # Known typed patterns still win exact-span/overlap ties because they were
        # appended earlier.
        for match in _STRONG_DOMAIN_CODE.finditer(text):
            value = match.group("value")
            if not any(ch.isdigit() for ch in value):
                continue
            context = text[max(0, match.start() - 100):match.start()]
            if not _CODE_CONTEXT.search(context):
                continue
            candidates.append(
                (match.start(), match.end(), value, "UNKNOWN_SENSITIVE", "domain_specific_identifier")
            )

        # Longest match wins on overlap. Stable sort preserves pattern priority for
        # exact-span ties (for example DOB before DATE).
        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        accepted: list[tuple[int, int, str, str, str]] = []
        for item in candidates:
            start, end, *_ = item
            if any(not (end <= a or start >= b) for a, b, *_ in accepted):
                continue
            accepted.append(item)

        critical = {
            "API_KEY",
            "BEARER_TOKEN",
            "ACCESS_TOKEN",
            "PASSWORD",
            "CONNECTION_STRING",
            "SECRET",
        }

        return [
            AnalyzedEntity(
                id=f"B{i}",
                type=entity_type,
                role=role,
                value=value,
                sensitive=True,
                sensitivity="critical" if entity_type in critical else "high",
                confidence=1.0,
                start=start,
                end=end,
                source="backstop",
            )
            for i, (start, end, value, entity_type, role) in enumerate(accepted, start=1)
        ]
