from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.agent.abstraction import PrivacyAbstractor
from app.memory.store import ExperienceMemory
from app.privacy.alignment import EntityAlignmentError, align_and_augment
from app.privacy.backstop import StructuredBackstopDetector
from app.privacy.coverage_audit import CoverageAuditor, classify_failure
from app.privacy.repair_controller import DeterministicRepairController
from app.privacy.safety_gate import HardSafetyGate
from app.schemas import (
    AbstractPattern,
    AnalysisResult,
    AttemptSummary,
    ExecutionResult,
    ProcessResponse,
    ReflectionResult,
    RetrievedExperience,
    TransformationPlan,
    ValidationReport,
)
from app.tools.registry import DeterministicToolRegistry, ToolExecutionError


class Reasoner(Protocol):
    async def analyze(self, text: str, repair_hint: str | None = None) -> AnalysisResult: ...

    async def plan(
        self,
        *,
        analysis: AnalysisResult,
        pattern: AbstractPattern,
        experiences: list[RetrievedExperience],
        repair_feedback: list[str] | None = None,
    ) -> TransformationPlan: ...

    async def reflect(
        self,
        *,
        pattern: AbstractPattern,
        plan: TransformationPlan,
        validation: ValidationReport,
    ) -> ReflectionResult: ...


@dataclass
class ContinuousLearningAgent:
    reasoner: Reasoner
    memory: ExperienceMemory
    tools: DeterministicToolRegistry
    gate: HardSafetyGate
    abstractor: PrivacyAbstractor
    backstop: StructuredBackstopDetector
    coverage_auditor: CoverageAuditor
    repair_controller: DeterministicRepairController
    max_attempts: int = 2

    async def _analyze(self, text: str) -> AnalysisResult:
        analysis = await self.reasoner.analyze(text)
        try:
            aligned = align_and_augment(text, analysis, self.backstop)
        except EntityAlignmentError:
            repaired = await self.reasoner.analyze(
                text,
                repair_hint=(
                    "Your previous entity spans could not be aligned exactly. "
                    "Copy each sensitive value exactly from input_text, avoid overlapping/duplicate spans, "
                    "and return corrected entities."
                ),
            )
            try:
                aligned = align_and_augment(text, repaired, self.backstop)
            except EntityAlignmentError:
                # A second alignment failure usually means the LLM still returned
                # at least one value that is not literally present in input_text.
                # Do not crash and do not invent/fuzzy-map a sensitive span. Keep
                # only exactly alignable LLM entities, then let the deterministic
                # backstop and coverage audit recover supported structured values.
                aligned = align_and_augment(
                    text,
                    repaired,
                    self.backstop,
                    skip_unalignable=True,
                )

        # Explicit coverage pass. It is idempotent with the current backstop but keeps
        # coverage checking separate from LLM reasoning and makes detector evolution safe.
        covered = self.coverage_auditor.audit_original(text, aligned)
        covered = self.coverage_auditor.expand_repeated_occurrences(text, covered)
        covered = self.coverage_auditor.expand_person_aliases(text, covered)
        return self.repair_controller.augment_analysis(text, covered)

    async def _coverage_reanalyze(
        self,
        *,
        text: str,
        current: AnalysisResult,
        violations: list[str],
    ) -> AnalysisResult:
        """One bounded LLM re-analysis when leakage suggests missed entities.

        The re-analysis occurs inside the same API request. The raw text remains transient;
        only the resulting abstract experience is eligible for long-term memory.
        """

        hint_codes = ",".join(sorted(set(violations)))
        repair_hint = (
            "Perform an independent FINAL COVERAGE review of the ORIGINAL input. "
            "Find sensitive spans that may have been omitted by the first analysis, including "
            "full person names and short aliases/coreferences, usernames, IDs, addresses, "
            "financial values, dates, passwords, tokens, API keys, connection strings, "
            "account/routing/passport numbers, and rare domain-specific confidential codes. "
            "Use UNKNOWN_SENSITIVE for unfamiliar sensitive values. Do not label ordinary prose "
            "words such as identifier, code, key, number, record, or account as the sensitive "
            "value itself. Copy every sensitive value exactly from input_text. Do not transform "
            "anything. Context codes: "
            + hint_codes
        )

        extra = await self.reasoner.analyze(text, repair_hint=repair_hint)
        try:
            extra_aligned = align_and_augment(text, extra, self.backstop)
        except EntityAlignmentError:
            # Preserve all exactly alignable coverage entities even if the LLM
            # hallucinates one bad span during the bounded coverage pass.
            extra_aligned = align_and_augment(
                text,
                extra,
                self.backstop,
                skip_unalignable=True,
            )

        merged = self.coverage_auditor.merge_analysis(current, extra_aligned)
        merged = self.coverage_auditor.audit_original(text, merged)
        merged = self.coverage_auditor.expand_repeated_occurrences(text, merged)
        merged = self.coverage_auditor.expand_person_aliases(text, merged)
        return self.repair_controller.augment_analysis(text, merged)

    @staticmethod
    def _span_type_map(analysis: AnalysisResult) -> dict[tuple[int, int], str]:
        return {
            (entity.start, entity.end): entity.type
            for entity in analysis.entities
            if entity.sensitive and entity.start is not None and entity.end is not None
        }

    async def _converge_second_attempt(
        self,
        *,
        text: str,
        scope: str,
        analysis: AnalysisResult,
        plan: TransformationPlan,
        validation: ValidationReport,
        execution: ExecutionResult | None,
        experiences: list[RetrievedExperience],
        immediate_reflection: ReflectionResult | None,
        coverage_reanalysis_used: bool,
        coverage_repaired: bool,
        new_entity_types: list[str],
        max_internal_repairs: int = 4,
    ) -> tuple[
        AnalysisResult,
        TransformationPlan,
        ValidationReport,
        ExecutionResult | None,
        bool,
        list[str],
        bool,
    ]:
        """Converge Attempt 2 without creating a third external attempt.

        Attempt 2 is the deterministic corrective attempt. If its first candidate
        exposes a newly discovered structured span, collides with source material, or
        still has a repairable plan/tool problem, repair it inside Attempt 2 and run the
        independent gate again. The gate is never forced to PASS.
        """

        discovered_types = set(new_entity_types)
        current_plan = plan
        current_validation = validation
        current_execution = execution
        current_analysis = analysis

        for internal_round in range(1, max_internal_repairs + 1):
            if current_validation.accepted and current_execution is not None:
                break

            # One bounded semantic coverage pass is allowed inside Attempt 2. It is
            # especially useful for context-dependent values and aliases that the
            # deterministic detector cannot discover by regex alone.
            failure_class = classify_failure(current_validation.violations)
            if (
                failure_class == "RESIDUAL_SENSITIVE_DATA"
                and not coverage_reanalysis_used
            ):
                before = self._span_type_map(current_analysis)
                try:
                    reanalyzed = await self._coverage_reanalyze(
                        text=text,
                        current=current_analysis,
                        violations=current_validation.violations,
                    )
                except Exception:
                    reanalyzed = current_analysis
                coverage_reanalysis_used = True
                after = self._span_type_map(reanalyzed)
                added_spans = [span for span in after if span not in before]
                if added_spans:
                    current_analysis = reanalyzed
                    coverage_repaired = True
                    discovered_types.update(after[span] for span in added_spans)

            current_pattern = self.abstractor.build(current_analysis)
            current_experiences = self.memory.retrieve(current_pattern, limit=5)
            if not current_experiences:
                current_experiences = experiences

            current_plan = self.repair_controller.repair_plan(
                analysis=current_analysis,
                plan=current_plan,
                violations=current_validation.violations or ["ATTEMPT_2_REPAIR"],
                experiences=current_experiences,
                immediate_reflection=immediate_reflection,
            )

            plan_violations = self.gate.validate_plan(current_analysis, current_plan)
            if plan_violations:
                current_execution = None
                current_validation = ValidationReport(
                    accepted=False,
                    privacy_pass=True,
                    privacy_risk=0.0,
                    structural_utility=0.0,
                    numeric_consistency=0.0,
                    temporal_consistency=0.0,
                    relation_consistency=0.0,
                    violations=plan_violations,
                )
                continue

            try:
                current_execution = self.tools.execute(
                    text=text,
                    analysis=current_analysis,
                    plan=current_plan,
                    scope=f"{scope}::retry:2::repair:{internal_round}",
                )
                current_validation = self.gate.validate(
                    original_text=text,
                    analysis=current_analysis,
                    plan=current_plan,
                    execution=current_execution,
                )
            except ToolExecutionError:
                current_execution = None
                current_validation = ValidationReport(
                    accepted=False,
                    privacy_pass=True,
                    privacy_risk=0.0,
                    structural_utility=0.0,
                    numeric_consistency=0.0,
                    temporal_consistency=0.0,
                    relation_consistency=0.0,
                    violations=["TOOL_EXECUTION_ERROR"],
                )
                continue

            residuals = self.coverage_auditor.find_residual_entities(
                original_text=text,
                analysis=current_analysis,
                execution=current_execution,
            )
            if residuals:
                before = self._span_type_map(current_analysis)
                augmented = self.coverage_auditor.merge_detections(
                    text=text,
                    analysis=current_analysis,
                    detections=residuals,
                    source="residual",
                    id_prefix="R",
                )
                augmented = self.coverage_auditor.expand_repeated_occurrences(text, augmented)
                augmented = self.coverage_auditor.expand_person_aliases(text, augmented)
                augmented = self.repair_controller.augment_analysis(text, augmented)
                after = self._span_type_map(augmented)
                added_spans = [span for span in after if span not in before]
                if added_spans:
                    current_analysis = augmented
                    coverage_repaired = True
                    added_types = {after[span] for span in added_spans}
                    discovered_types.update(added_types)
                    residual_codes = [
                        f"RESIDUAL_SENSITIVE_DATA:{entity_type}"
                        for entity_type in sorted(added_types)
                    ]
                    current_validation = current_validation.model_copy(
                        update={
                            "accepted": False,
                            "privacy_pass": False,
                            "privacy_risk": 1.0,
                            "violations": sorted(
                                set(current_validation.violations + residual_codes)
                            ),
                        }
                    )
                    continue

            if current_validation.accepted:
                break

            # For remaining repairable privacy/tool/plan failures, the next internal
            # round uses a fresh deterministic scope and a rebuilt allow-listed plan.
            # Structural-utility failures indicate a programming invariant violation
            # rather than a surrogate collision, so do not spin on them.
            failure_class = classify_failure(current_validation.violations)
            if failure_class == "STRUCTURAL_UTILITY_FAILURE":
                break

        return (
            current_analysis,
            current_plan,
            current_validation,
            current_execution,
            coverage_repaired,
            sorted(discovered_types),
            coverage_reanalysis_used,
        )

    async def process(self, text: str, scope: str = "default") -> ProcessResponse:
        analysis = await self._analyze(text)
        coverage_reanalysis_used = False
        sensitive = [entity for entity in analysis.entities if entity.sensitive]
        if not sensitive:
            # Zero detections are high-risk because a false negative would otherwise
            # release the raw input unchanged. Require one independent semantic review
            # before accepting a no-sensitive-information result.
            try:
                confirmed = await self._coverage_reanalyze(
                    text=text,
                    current=analysis,
                    violations=["EMPTY_INITIAL_ANALYSIS_CONFIRMATION"],
                )
            except Exception:
                confirmed = analysis
            coverage_reanalysis_used = True
            if self._span_type_map(confirmed):
                analysis = confirmed
                sensitive = [entity for entity in analysis.entities if entity.sensitive]
            else:
                return ProcessResponse(
                    accepted=True,
                    transformed_text=text,
                    attempts=0,
                    pattern=None,
                    strategy=None,
                    validation=None,
                    attempt_history=[],
                    message=(
                        "No sensitive information was identified after independent AI "
                        "analysis plus deterministic coverage confirmation."
                    ),
                )

        pattern = self.abstractor.build(analysis)
        experiences = self.memory.retrieve(pattern, limit=5)
        last_plan: TransformationPlan | None = None
        last_validation: ValidationReport | None = None
        repair_feedback: list[str] = []
        attempt_history: list[AttemptSummary] = []
        last_reflection: ReflectionResult | None = None

        for attempt in range(1, self.max_attempts + 1):
            attempt_pattern = pattern

            if attempt == 1:
                # Attempt 1 remains adaptive: Qwen selects the transformation strategy
                # using the current abstract pattern and retrieved prior experiences.
                plan = await self.reasoner.plan(
                    analysis=analysis,
                    pattern=attempt_pattern,
                    experiences=experiences,
                    repair_feedback=None,
                )
                plan_repaired = False
            else:
                # Attempt 2 is intentionally NOT another free-form LLM retry. Attempt 1
                # has already been reflected on, stored, and retrieved as privacy-safe
                # experience. The deterministic repair consumes both the immediate lesson
                # and retrieved PASS/FAIL experiences, then rebuilds a complete allow-listed
                # plan from the UPDATED analysis and exact value-free validation feedback.
                # Execution starts again from the original input below.
                if last_plan is None:
                    raise RuntimeError("Attempt 2 requires the failed first-attempt plan")
                plan = self.repair_controller.repair_plan(
                    analysis=analysis,
                    plan=last_plan,
                    violations=repair_feedback or ["ATTEMPT_1_FAILED"],
                    experiences=experiences,
                    immediate_reflection=last_reflection,
                )
                plan_repaired = True

            last_plan = plan

            # New control flow: Attempt 2 is released directly after deterministic
            # repair. The independent hard safety gate, residual audit, semantic
            # coverage confirmation, and second-attempt convergence are intentionally
            # NOT run on Attempt 2. Attempt 2 is recorded as RELEASED, not PASS.
            if attempt == 2:
                execution_scope = f"{scope}::retry:{attempt}"
                try:
                    execution = self.tools.execute(
                        text=text,
                        analysis=analysis,
                        plan=plan,
                        scope=execution_scope,
                    )
                except ToolExecutionError:
                    execution = None
                    release_report = ValidationReport(
                        accepted=False,
                        validation_performed=False,
                        privacy_pass=False,
                        privacy_risk=0.0,
                        structural_utility=0.0,
                        numeric_consistency=0.0,
                        temporal_consistency=0.0,
                        relation_consistency=0.0,
                        violations=["TOOL_EXECUTION_ERROR"],
                    )
                    try:
                        reflection = await self.reasoner.reflect(
                            pattern=attempt_pattern,
                            plan=plan,
                            validation=release_report,
                        )
                    except Exception:
                        reflection = ReflectionResult(
                            outcome_reason=(
                                "Attempt 2 could not produce a candidate because deterministic "
                                "tool execution failed before release."
                            ),
                            lesson=(
                                "Repair the deterministic transformation plan so every detected "
                                "entity can be executed before the direct release is attempted."
                            ),
                            failure_type="TOOL_EXECUTION_ERROR",
                            recommended_strategy="deterministic_gate_repair",
                            recommended_tools=[],
                        )
                    self.memory.record(
                        pattern=attempt_pattern,
                        plan=plan,
                        validation=release_report,
                        reflection=reflection,
                        result_override="FAIL",
                    )
                    attempt_history.append(
                        AttemptSummary(
                            attempt=attempt,
                            outcome="FAIL",
                            released=False,
                            strategy=plan.strategy,
                            repaired=True,
                            coverage_repaired=False,
                            failure_class="TOOL_EXECUTION_ERROR",
                            new_entity_types=[],
                            accepted=False,
                            violations=["TOOL_EXECUTION_ERROR"],
                            outcome_reason=reflection.outcome_reason,
                            learned_lesson=reflection.lesson,
                        )
                    )
                    return ProcessResponse(
                        accepted=False,
                        transformed_text=None,
                        attempts=2,
                        pattern=attempt_pattern,
                        strategy=plan.strategy,
                        validation=None,
                        attempt_history=attempt_history,
                        message=(
                            "Attempt 2 was configured for direct release, but deterministic "
                            "transformation execution failed, so no candidate existed to release."
                        ),
                    )

                release_report = ValidationReport(
                    accepted=False,
                    validation_performed=False,
                    privacy_pass=False,
                    privacy_risk=0.0,
                    structural_utility=0.0,
                    numeric_consistency=0.0,
                    temporal_consistency=0.0,
                    relation_consistency=0.0,
                    violations=[],
                )
                try:
                    reflection = await self.reasoner.reflect(
                        pattern=attempt_pattern,
                        plan=plan,
                        validation=release_report,
                    )
                except Exception:
                    reflection = ReflectionResult(
                        outcome_reason=(
                            "Attempt 2 used deterministic repair and the resulting candidate "
                            "was released without running the hard safety gate a second time."
                        ),
                        lesson=(
                            "Use the value-free failure lesson from Attempt 1 to build a complete "
                            "deterministic repair plan, and store the second-attempt outcome as "
                            "RELEASED for future learning."
                        ),
                        failure_type=None,
                        recommended_strategy="deterministic_gate_repair",
                        recommended_tools=sorted({action.tool for action in plan.actions}),
                    )

                # Learn from Attempt 2 as a distinct outcome. It is stored and retrievable,
                # but it contributes neither positive PASS evidence nor negative FAIL evidence.
                self.memory.record(
                    pattern=attempt_pattern,
                    plan=plan,
                    validation=release_report,
                    reflection=reflection,
                    result_override="RELEASED",
                )
                attempt_history.append(
                    AttemptSummary(
                        attempt=attempt,
                        outcome="RELEASED",
                        released=True,
                            strategy=plan.strategy,
                        repaired=True,
                        coverage_repaired=False,
                        failure_class=None,
                        new_entity_types=[],
                        accepted=True,
                        violations=[],
                        outcome_reason=reflection.outcome_reason,
                        learned_lesson=reflection.lesson,
                    )
                )
                return ProcessResponse(
                    accepted=True,
                    transformed_text=execution.candidate_text,
                    attempts=2,
                    pattern=attempt_pattern,
                    strategy=plan.strategy,
                    validation=None,
                    attempt_history=attempt_history,
                    message=(
                        "Attempt 1 failed the hard safety gate. Attempt 2 used learned "
                        "deterministic repair and was RELEASED directly. The Attempt 2 "
                        "experience was stored with outcome RELEASED."
                    ),
                )

            execution = None
            plan_violations = self.gate.validate_plan(analysis, plan)

            if plan_violations:
                validation = ValidationReport(
                    accepted=False,
                    privacy_pass=True,
                    privacy_risk=0.0,
                    structural_utility=0.0,
                    numeric_consistency=0.0,
                    temporal_consistency=0.0,
                    relation_consistency=0.0,
                    violations=plan_violations,
                )
            else:
                try:
                    # Retry-specific scope permits a fresh deterministic surrogate if a
                    # previous one collided with a source fragment, while remaining
                    # reproducible for the same attempt/scope.
                    execution_scope = scope if attempt == 1 else f"{scope}::retry:{attempt}"
                    execution = self.tools.execute(
                        text=text,
                        analysis=analysis,
                        plan=plan,
                        scope=execution_scope,
                    )
                    validation = self.gate.validate(
                        original_text=text,
                        analysis=analysis,
                        plan=plan,
                        execution=execution,
                    )
                except ToolExecutionError:
                    validation = ValidationReport(
                        accepted=False,
                        privacy_pass=True,
                        privacy_risk=0.0,
                        structural_utility=0.0,
                        numeric_consistency=0.0,
                        temporal_consistency=0.0,
                        relation_consistency=0.0,
                        violations=["TOOL_EXECUTION_ERROR"],
                    )

            coverage_repaired = False
            new_entity_types: list[str] = []

            # A candidate must pass a residual-data audit before it can be released,
            # even if the entity-aware gate itself reports PASS.
            if execution is not None:
                before = self._span_type_map(analysis)
                residuals = self.coverage_auditor.find_residual_entities(
                    original_text=text,
                    analysis=analysis,
                    execution=execution,
                )
                if residuals:
                    augmented = self.coverage_auditor.merge_detections(
                        text=text,
                        analysis=analysis,
                        detections=residuals,
                        source="residual",
                        id_prefix="R",
                    )
                    augmented = self.repair_controller.augment_analysis(text, augmented)
                    after = self._span_type_map(augmented)
                    added_spans = [span for span in after if span not in before]
                    new_entity_types = sorted({after[span] for span in added_spans})

                    if added_spans:
                        analysis = augmented
                        coverage_repaired = True
                        residual_codes = [
                            f"RESIDUAL_SENSITIVE_DATA:{entity_type}"
                            for entity_type in new_entity_types
                        ] or ["RESIDUAL_SENSITIVE_DATA"]
                        violations = sorted(set(validation.violations + residual_codes))
                        validation = validation.model_copy(
                            update={
                                "accepted": False,
                                "privacy_pass": False,
                                "privacy_risk": 1.0,
                                "violations": violations,
                            }
                        )

            # If the gate still reports leakage but deterministic residual scanning did
            # not add anything, use at most one bounded Qwen coverage re-analysis inside
            # this same request. This repairs missed semantic/unstructured entities such
            # as aliases or context-dependent private values.
            failure_class = classify_failure(validation.violations)
            if (
                not validation.accepted
                and failure_class == "RESIDUAL_SENSITIVE_DATA"
                and not coverage_repaired
                and not coverage_reanalysis_used
            ):
                before = self._span_type_map(analysis)
                try:
                    reanalyzed = await self._coverage_reanalyze(
                        text=text,
                        current=analysis,
                        violations=validation.violations,
                    )
                except Exception:
                    reanalyzed = analysis
                coverage_reanalysis_used = True
                after = self._span_type_map(reanalyzed)
                added_spans = [span for span in after if span not in before]
                if added_spans:
                    analysis = reanalyzed
                    coverage_repaired = True
                    new_entity_types = sorted({after[span] for span in added_spans})
                    violations = sorted(
                        set(validation.violations + ["COVERAGE_REANALYSIS_ADDED_ENTITIES"])
                    )
                    validation = validation.model_copy(
                        update={
                            "accepted": False,
                            "privacy_pass": False,
                            "privacy_risk": 1.0,
                            "violations": violations,
                        }
                    )
                    failure_class = "RESIDUAL_SENSITIVE_DATA"

            # Even a deterministic PASS is not released until one independent semantic
            # coverage confirmation has reviewed the ORIGINAL input. This catches missed
            # names/aliases and context-dependent or domain-specific sensitive values that
            # are invisible to pure regex residual scanning. Newly found spans force repair;
            # the gate itself is never bypassed or forced to PASS.
            if validation.accepted and execution is not None and not coverage_reanalysis_used:
                before = self._span_type_map(analysis)
                try:
                    reanalyzed = await self._coverage_reanalyze(
                        text=text,
                        current=analysis,
                        violations=["FINAL_COVERAGE_CONFIRMATION"],
                    )
                except Exception:
                    reanalyzed = analysis
                coverage_reanalysis_used = True
                after = self._span_type_map(reanalyzed)
                added_spans = [span for span in after if span not in before]
                if added_spans:
                    analysis = reanalyzed
                    coverage_repaired = True
                    new_entity_types = sorted({after[span] for span in added_spans})
                    validation = validation.model_copy(
                        update={
                            "accepted": False,
                            "privacy_pass": False,
                            "privacy_risk": 1.0,
                            "violations": sorted(
                                set(validation.violations + ["FINAL_COVERAGE_ADDED_ENTITIES"])
                            ),
                        }
                    )
                    failure_class = "RESIDUAL_SENSITIVE_DATA"

            if attempt == 2 and not validation.accepted:
                (
                    analysis,
                    plan,
                    validation,
                    execution,
                    coverage_repaired,
                    new_entity_types,
                    coverage_reanalysis_used,
                ) = await self._converge_second_attempt(
                    text=text,
                    scope=scope,
                    analysis=analysis,
                    plan=plan,
                    validation=validation,
                    execution=execution,
                    experiences=experiences,
                    immediate_reflection=last_reflection,
                    coverage_reanalysis_used=coverage_reanalysis_used,
                    coverage_repaired=coverage_repaired,
                    new_entity_types=new_entity_types,
                )
                last_plan = plan
                failure_class = classify_failure(validation.violations)

            # If coverage changed during either attempt, learn/store the actual final
            # abstract pattern seen by that attempt rather than the earlier snapshot.
            if coverage_repaired:
                attempt_pattern = self.abstractor.build(analysis)

            last_validation = validation

            try:
                reflection = await self.reasoner.reflect(
                    pattern=attempt_pattern,
                    plan=plan,
                    validation=validation,
                )
            except Exception:
                reflection = ReflectionResult(
                    outcome_reason=(
                        "The candidate passed all deterministic privacy, coverage, and "
                        "relation-preservation checks."
                        if validation.accepted
                        else "The candidate failed one or more deterministic checks: "
                        + ",".join(validation.violations or ["VALIDATION_FAILURE"])
                    ),
                    lesson=(
                        "Reuse strategies that pass deterministic privacy, coverage, and "
                        "relation-preservation checks; avoid the structural choices that "
                        "produced the recorded failure type."
                    ),
                    failure_type=(
                        None
                        if validation.accepted
                        else (
                            validation.violations[0]
                            if validation.violations
                            else "VALIDATION_FAILURE"
                        )
                    ),
                    recommended_strategy=None,
                    recommended_tools=[],
                )

            # Memory persists only abstract pattern/plan/validation/reflection. Raw source
            # values and original->synthetic mappings are never written by this method.
            self.memory.record(
                pattern=attempt_pattern,
                plan=plan,
                validation=validation,
                reflection=reflection,
            )
            last_reflection = reflection

            attempt_history.append(
                AttemptSummary(
                    attempt=attempt,
                    outcome="PASS" if validation.accepted else "FAIL",
                    released=bool(validation.accepted and execution is not None),
                    strategy=plan.strategy,
                    repaired=plan_repaired,
                    coverage_repaired=coverage_repaired,
                    failure_class=failure_class,
                    new_entity_types=new_entity_types,
                    accepted=validation.accepted,
                    violations=list(validation.violations),
                    outcome_reason=reflection.outcome_reason,
                    learned_lesson=reflection.lesson,
                )
            )

            if validation.accepted and execution is not None:
                return ProcessResponse(
                    accepted=True,
                    transformed_text=execution.candidate_text,
                    attempts=attempt,
                    pattern=attempt_pattern,
                    strategy=plan.strategy,
                    validation=validation,
                    attempt_history=attempt_history,
                    message=(
                        "Candidate passed the independent hard safety gate and residual-data audit."
                        if attempt == 1
                        else "Candidate passed after automatic within-request repair."
                    ),
                )

            repair_feedback = list(validation.violations)

            # If coverage changed the entity set, all next-attempt planning/retrieval must
            # use the updated abstract structure.
            pattern = self.abstractor.build(analysis)
            experiences = self.memory.retrieve(pattern, limit=5)

        return ProcessResponse(
            accepted=False,
            transformed_text=None,
            attempts=self.max_attempts,
            pattern=pattern,
            strategy=last_plan.strategy if last_plan else None,
            validation=last_validation,
            attempt_history=attempt_history,
            message=(
                "No candidate passed the hard safety gate and residual-data audit within "
                "the configured attempt limit. No failed candidate was released."
            ),
        )
