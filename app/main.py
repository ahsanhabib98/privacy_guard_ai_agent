from __future__ import annotations

from app.agent.abstraction import PrivacyAbstractor
from app.agent.service import ContinuousLearningAgent
from app.config import settings
from app.llm.client import QwenOllamaClient
from app.llm.reasoner import LocalQwenReasoner
from app.memory.store import ExperienceMemory
from app.privacy.backstop import StructuredBackstopDetector
from app.privacy.coverage_audit import CoverageAuditor
from app.privacy.repair_controller import DeterministicRepairController
from app.privacy.safety_gate import HardSafetyGate
from app.tools.registry import DeterministicToolRegistry


def create_agent() -> ContinuousLearningAgent:
    settings.validate()

    client = QwenOllamaClient(settings)
    reasoner = LocalQwenReasoner(client)

    memory = ExperienceMemory(settings.memory_db)

    tools = DeterministicToolRegistry(
        settings.privacy_hmac_secret
    )

    gate = HardSafetyGate(tools)

    repair_controller = DeterministicRepairController(
        tools
    )

    backstop = StructuredBackstopDetector()

    coverage_auditor = CoverageAuditor(
        backstop
    )

    agent = ContinuousLearningAgent(
        reasoner=reasoner,
        memory=memory,
        tools=tools,
        gate=gate,
        abstractor=PrivacyAbstractor(),
        backstop=backstop,
        coverage_auditor=coverage_auditor,
        repair_controller=repair_controller,
        max_attempts=settings.max_attempts,
    )

    return agent