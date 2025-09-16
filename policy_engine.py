from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from llm import query_model


class PlanningFormatError(Exception):
    """Raised when the LLM response for the policy engine does not match the contract."""


@dataclass
class SymptomAssessment:
    name: str
    weight: float
    expected_raw: str
    observation_raw: str
    evidence: str = ""
    expected_state: str = field(init=False)
    observed_state: str = field(init=False)

    def __post_init__(self) -> None:
        self.expected_state = _normalize_presence(self.expected_raw)
        self.observed_state = _normalize_presence(self.observation_raw)

    def is_supported(self) -> bool:
        return (
            self.expected_state in {"present", "absent"}
            and self.observed_state in {"present", "absent"}
            and self.expected_state == self.observed_state
        )

    def is_contradictory(self) -> bool:
        return (
            self.expected_state in {"present", "absent"}
            and self.observed_state in {"present", "absent"}
            and self.expected_state != self.observed_state
        )

    def is_unknown(self) -> bool:
        return self.observed_state == "unknown" or self.expected_state == "unknown"


@dataclass
class DiseaseHypothesis:
    name: str
    summary: str
    key_symptoms: List[SymptomAssessment]
    total_weight: float = 0.0
    support_weight: float = 0.0
    contradiction_weight: float = 0.0
    unknown_weight: float = 0.0
    coverage: float = 0.0
    confidence: float = 0.0

    def compute_metrics(self) -> None:
        totals = 0.0
        support = 0.0
        contradiction = 0.0
        for symptom in self.key_symptoms:
            weight = max(symptom.weight, 0.0)
            if weight == 0.0:
                continue
            totals += weight
            if symptom.is_supported():
                support += weight
            elif symptom.is_contradictory():
                contradiction += weight
        self.total_weight = totals
        self.support_weight = support
        self.contradiction_weight = contradiction
        self.unknown_weight = max(0.0, totals - support - contradiction)
        if totals > 0.0:
            raw = (support - contradiction) / totals
            self.confidence = max(0.0, min(1.0, (raw + 1.0) / 2.0))
            self.coverage = (support + contradiction) / totals
        else:
            self.confidence = 0.0
            self.coverage = 0.0

    def top_unknowns(self, limit: int = 3) -> List[SymptomAssessment]:
        unknowns = [sym for sym in self.key_symptoms if sym.is_unknown()]
        unknowns.sort(key=lambda sym: sym.weight, reverse=True)
        return unknowns[:limit]

    def contradictory_features(self, limit: int = 3) -> List[SymptomAssessment]:
        contradictions = [sym for sym in self.key_symptoms if sym.is_contradictory()]
        contradictions.sort(key=lambda sym: sym.weight, reverse=True)
        return contradictions[:limit]


@dataclass
class PlannerDecision:
    hypotheses: List[DiseaseHypothesis]
    finish_threshold: float
    close_gap: float
    should_finish: bool
    chosen_diagnosis: Optional[str]
    finish_reason: Optional[str]
    close_competitors: List[DiseaseHypothesis] = field(default_factory=list)
    discriminators: List[Dict[str, Any]] = field(default_factory=list)
    action_summary: Optional[str] = None

    def top_hypothesis(self) -> Optional[DiseaseHypothesis]:
        return self.hypotheses[0] if self.hypotheses else None


REASONER_SYSTEM_PROMPT = (
    "You are an expert clinical reasoning planner. Maintain a weighted differential diagnosis, "
    "tracking how transcript evidence supports or contradicts each candidate. Always return valid JSON "
    "that matches the requested schema."
)

DISCRIMINATOR_SYSTEM_PROMPT = (
    "You help a clinician focus follow-up questions or tests that rapidly discriminate between close diagnoses. "
    "Always return valid JSON matching the requested schema."
)


class DifferentialDiagnosisEngine:
    """LLM-powered differential engine that turns policy guidance into structured plans."""

    def __init__(
        self,
        *,
        llm_model: str,
        policy: Optional[Dict[str, Any]] = None,
        finish_threshold: float = 0.7,
        case_overview: str = "",
        patient_overview: str = "",
    ) -> None:
        self.llm_model = llm_model
        self.policy = policy or {}
        settings = (self.policy.get("differential_settings") or {})
        requested_min = int(_coerce_float(settings.get("min_symptoms_per_disease", 4), 4))
        requested_min = max(3, requested_min)
        self.min_symptoms_requested = requested_min
        self.min_symptoms = requested_min
        self._min_symptoms_active = requested_min
        self.max_differentials = int(_coerce_float(settings.get("max_differentials", 5), 5))
        self.max_differentials = max(3, self.max_differentials)
        self.min_differentials = min(3, self.max_differentials)
        self.close_gap = float(_coerce_float(settings.get("confidence_close_threshold", 0.12), 0.12))
        self.finish_threshold = float(finish_threshold)
        self.case_overview = (case_overview or "").strip()
        self.patient_overview = (patient_overview or "").strip()
        self.max_reasoner_retries = 3
        self.max_discriminator_retries = 2
        self.last_decision: Optional[PlannerDecision] = None
        self._pending_relaxation_hint: Optional[str] = None
        self._reduction_note: Optional[str] = None
        self._fallback_note: Optional[str] = None

    def analyze(self, agent_hist: str, latest_turn: str) -> PlannerDecision:
        context = self._compose_context(agent_hist or "", latest_turn or "")
        self._pending_relaxation_hint = None
        self._fallback_note = None
        try:
            hypotheses = self._run_reasoner(context)
        except Exception as err:
            decision = self._fallback_decision(str(err))
            self.last_decision = decision
            return decision
        decision = self._build_decision(hypotheses)
        if decision.close_competitors:
            decision.discriminators = self._run_discriminator(context, decision)
        decision.action_summary = self._build_action_summary(decision)
        self._maybe_restore_min_requirement()
        self.last_decision = decision
        return decision

    def render_guidance(self, decision: Optional[PlannerDecision] = None) -> str:
        decision = decision or self.last_decision
        if decision is None:
            return ""
        lines: List[str] = []
        lines.append(
            f"Finish threshold {decision.finish_threshold * 100:.0f}% | close gap <= {decision.close_gap:.2f}"
        )
        if self._reduction_note:
            lines.append(self._reduction_note)
        if self._fallback_note:
            lines.append(self._fallback_note)
        if decision.hypotheses:
            limit = min(4, len(decision.hypotheses))
            for idx in range(limit):
                hyp = decision.hypotheses[idx]
                lines.append(
                    f"{idx + 1}. {hyp.name}: conf {hyp.confidence:.2f}, coverage {hyp.coverage:.2f}, "
                    f"support {hyp.support_weight:.1f}, contra {hyp.contradiction_weight:.1f}"
                )
                unresolved = [sym.name for sym in hyp.top_unknowns(limit=2)]
                if unresolved:
                    lines.append(f"   unresolved: {', '.join(unresolved)}")
        if decision.should_finish and decision.chosen_diagnosis:
            reason = decision.finish_reason or "confidence threshold met"
            lines.append(f"Recommendation: move to diagnosis for {decision.chosen_diagnosis} ({reason}).")
        else:
            if decision.action_summary:
                lines.append(f"Next focus: {decision.action_summary}")
            if decision.close_competitors:
                competitors = ", ".join(hyp.name for hyp in decision.close_competitors[:3])
                lines.append(f"Close competitors: {competitors}")
        return "\n".join(lines)

    def _compose_context(self, agent_hist: str, latest_turn: str) -> str:
        parts: List[str] = []
        if self.case_overview:
            parts.append(f"Examiner objective: {self.case_overview}")
        if self.patient_overview:
            parts.append(f"Baseline patient information: {self.patient_overview}")
        trimmed_hist = agent_hist.strip()
        if trimmed_hist:
            parts.append("Transcript so far:\n" + trimmed_hist)
        latest = (latest_turn or "").strip()
        if latest and latest not in trimmed_hist:
            parts.append("Latest patient or test message:\n" + latest)
        return "\n\n".join(parts).strip()

    def _run_reasoner(self, context: str) -> List[DiseaseHypothesis]:
        while True:
            retry_hint = self._pending_relaxation_hint
            self._pending_relaxation_hint = None
            for _ in range(self.max_reasoner_retries):
                prompt = self._build_reasoner_prompt(context, retry_hint)
                retry_hint = None
                raw = query_model(self.llm_model, prompt, REASONER_SYSTEM_PROMPT)
                data = _safe_load_json(raw)
                if data is None:
                    retry_hint = "Output was not valid JSON."
                    continue
                try:
                    hypotheses = self._extract_hypotheses(data)
                except PlanningFormatError as err:
                    retry_hint = str(err)
                    continue
                if hypotheses:
                    return hypotheses
                retry_hint = "No valid differential diagnoses were produced."
            if self._min_symptoms_active > 3:
                self._min_symptoms_active -= 1
                self._pending_relaxation_hint = (
                    "Requirement adjusted: include at least {} key symptoms for every diagnosis. You may list additional distinguishing findings with observation marked 'unknown' whenever evidence is missing."
                    .format(self._min_symptoms_active)
                )
                if self._min_symptoms_active < self.min_symptoms_requested:
                    self._reduction_note = (
                        f"Minimum key-symptom requirement relaxed to {self._min_symptoms_active} (requested {self.min_symptoms_requested})."
                    )
                continue
            raise RuntimeError("Differential planner failed to produce structured differential output.")

    def _build_reasoner_prompt(self, context: str, retry_hint: Optional[str]) -> str:
        ctx = context if context else "No dialogue yet. Use the baseline information to seed the differential."
        schema = json.dumps(
            {
                "differentials": [
                    {
                        "disease": "Diagnosis name",
                        "summary": "One sentence justification that cites the transcript",
                        "key_symptoms": [
                            {
                                "name": "Finding or symptom",
                                "weight": 4,
                                "expected": "present",
                                "observation": "present",
                                "evidence": "Quote or summary from the transcript",
                            }
                        ],
                    }
                ]
            },
            indent=2,
        )
        active_min = self._min_symptoms_active
        prompt = (
            f"Case context:\n{ctx}\n\n"
            f"Tasks:\n"
            f"1. List between {self.min_differentials} and {self.max_differentials} plausible diagnoses that explain the findings."
            f"\n2. For each diagnosis, include at least {active_min} high-yield distinguishing symptoms or test findings."
            "\n3. Assign an integer weight from 1-5 for each finding based on diagnostic value."
            "\n4. Set expected to 'present' or 'absent' to reflect whether the disease predicts that finding."
            "\n5. Set observation to 'present', 'absent', or 'unknown' based strictly on the transcript and test results; when evidence is missing, still list the distinguishing finding and mark observation 'unknown'."
            "\n6. Provide a one sentence summary for each diagnosis explaining the rationale."
            "\n\nReturn ONLY JSON following this schema:\n"
            f"{schema}\n"
            "Do not include any additional narration."
        )
        if retry_hint:
            prompt += f"\nCorrection needed: {retry_hint}"
        return prompt

    def _extract_hypotheses(self, data: Dict[str, Any]) -> List[DiseaseHypothesis]:
        differentials = data.get("differentials")
        if not isinstance(differentials, list):
            raise PlanningFormatError("Missing 'differentials' list in JSON response.")
        if len(differentials) < self.min_differentials:
            raise PlanningFormatError(
                f"At least {self.min_differentials} diagnoses are required; received {len(differentials)}."
            )
        hypotheses: List[DiseaseHypothesis] = []
        problems: List[str] = []
        for entry in differentials[: self.max_differentials]:
            disease = _clean_text(entry.get("disease")).strip()
            if not disease:
                problems.append("Encountered diagnosis entry without a name.")
                continue
            raw_symptoms = entry.get("key_symptoms")
            if not isinstance(raw_symptoms, list):
                problems.append(f"{disease}: 'key_symptoms' must be a list.")
                continue
            symptoms: List[SymptomAssessment] = []
            for symptom_entry in raw_symptoms:
                name = _clean_text(symptom_entry.get("name")).strip()
                if not name:
                    continue
                weight = max(0.5, _coerce_float(symptom_entry.get("weight"), 1.0))
                expected = _clean_text(symptom_entry.get("expected")).strip()
                observation = _clean_text(symptom_entry.get("observation")).strip()
                evidence = _clean_text(
                    symptom_entry.get("evidence") or symptom_entry.get("rationale")
                ).strip()
                symptoms.append(
                    SymptomAssessment(
                        name=name,
                        weight=weight,
                        expected_raw=expected,
                        observation_raw=observation,
                        evidence=evidence,
                    )
                )
            if len(symptoms) < self._min_symptoms_active:
                problems.append(
                    f"{disease} must include at least {self._min_symptoms_active} key symptoms; received {len(symptoms)}."
                )
                continue
            summary = _clean_text(entry.get("summary") or entry.get("rationale")).strip()
            hypothesis = DiseaseHypothesis(name=disease, summary=summary, key_symptoms=symptoms)
            hypothesis.compute_metrics()
            hypotheses.append(hypothesis)
        if problems:
            raise PlanningFormatError("; ".join(problems))
        if not hypotheses:
            raise PlanningFormatError("Differential list was empty after validation.")
        return hypotheses

    def _build_decision(self, hypotheses: Sequence[DiseaseHypothesis]) -> PlannerDecision:
        ordered = sorted(hypotheses, key=lambda hyp: hyp.confidence, reverse=True)
        top = ordered[0]
        close = [hyp for hyp in ordered[1:] if top.confidence - hyp.confidence <= self.close_gap]
        should_finish = False
        chosen = None
        finish_reason = None
        if (
            top.confidence >= self.finish_threshold
            and top.coverage >= 0.6
            and top.support_weight >= top.contradiction_weight
            and not close
        ):
            margin = top.confidence - (ordered[1].confidence if len(ordered) > 1 else 0.0)
            should_finish = True
            chosen = top.name
            finish_reason = (
                f"confidence {top.confidence:.2f} >= {self.finish_threshold:.2f} with coverage {top.coverage:.2f} "
                f"and margin {margin:.2f}"
            )
        return PlannerDecision(
            hypotheses=list(ordered),
            finish_threshold=self.finish_threshold,
            close_gap=self.close_gap,
            should_finish=should_finish,
            chosen_diagnosis=chosen,
            finish_reason=finish_reason,
            close_competitors=close,
        )

    def _fallback_decision(self, error_message: str) -> PlannerDecision:
        detail = (error_message or "").strip().splitlines()[0]
        if len(detail) > 160:
            detail = detail[:157] + "..."
        summary = (
            "Reset to basics: ask an open-ended question about onset, timing, associated symptoms, and red flags before planning tests."
        )
        note = "Planner temporarily unavailable."
        if detail:
            note = f"Planner temporarily unavailable: {detail}"
        self._fallback_note = note
        return PlannerDecision(
            hypotheses=[],
            finish_threshold=self.finish_threshold,
            close_gap=self.close_gap,
            should_finish=False,
            chosen_diagnosis=None,
            finish_reason=None,
            close_competitors=[],
            discriminators=[],
            action_summary=summary,
        )



    def _run_discriminator(self, context: str, decision: PlannerDecision) -> List[Dict[str, Any]]:
        retry_hint = None
        for _ in range(self.max_discriminator_retries):
            prompt = self._build_discriminator_prompt(context, decision, retry_hint)
            raw = query_model(self.llm_model, prompt, DISCRIMINATOR_SYSTEM_PROMPT)
            data = _safe_load_json(raw)
            if data is None:
                retry_hint = "Output was not valid JSON."
                continue
            clarifications = data.get("priority_clarifications")
            if not isinstance(clarifications, list) or not clarifications:
                retry_hint = "Missing 'priority_clarifications' list."
                continue
            results: List[Dict[str, Any]] = []
            for item in clarifications:
                focus = _clean_text(item.get("focus") or item.get("symptom")).strip()
                if not focus:
                    continue
                action_type = _clean_text(item.get("action_type") or item.get("type") or "clarify").strip()
                reason = _clean_text(item.get("reason") or item.get("why")).strip()
                targets_raw = item.get("targets") or item.get("target_diseases") or []
                if isinstance(targets_raw, str):
                    targets = [_clean_text(targets_raw).strip()]
                else:
                    targets = [
                        _clean_text(target).strip()
                        for target in targets_raw
                        if _clean_text(target).strip()
                    ]
                results.append(
                    {
                        "focus": focus,
                        "action_type": action_type or "clarify",
                        "reason": reason,
                        "targets": targets,
                    }
                )
            if results:
                return results[:3]
            retry_hint = "Clarifications were missing required fields."
        return []

    def _build_discriminator_prompt(
        self,
        context: str,
        decision: PlannerDecision,
        retry_hint: Optional[str],
    ) -> str:
        top = decision.top_hypothesis()
        competitors = [hyp for hyp in [top] + decision.close_competitors if hyp is not None]
        summary_lines: List[str] = []
        for hyp in competitors:
            unresolved = [
                f"{sym.name} ({'expect present' if sym.expected_state == 'present' else 'expect absent'})"
                for sym in hyp.top_unknowns(limit=3)
            ]
            contradictions = [
                f"{sym.name} (expected {sym.expected_state}, observed {sym.observed_state})"
                for sym in hyp.contradictory_features(limit=2)
            ]
            summary_lines.append(
                f"{hyp.name}: unresolved {', '.join(unresolved) or 'none'}; contradictions {', '.join(contradictions) or 'none'}."
            )
        opposing = self._collect_opposing_expectations(competitors)
        ctx = context if context else "No dialogue yet."
        prompt = (
            f"Case context:\n{ctx}\n\n"
            f"Competing diagnoses: {', '.join(hyp.name for hyp in competitors)}.\n"
            f"Key gaps:\n" + ("\n".join(summary_lines) if summary_lines else "none") + "\n\n"
            f"Opposing expectations to prioritize:\n" + ("\n".join(opposing) if opposing else "none") + "\n\n"
            "Select the highest-yield clarifications (questions or tests) that quickly discriminate these diagnoses. "
            "Prefer evidence that would strongly support one disease while refuting another."
            "\nReturn ONLY JSON with this schema:\n"
            "{\n"
            "  \"priority_clarifications\": [\n"
            "    {\n"
            "      \"focus\": \"finding or test to clarify\",\n"
            "      \"action_type\": \"question\" or \"test\",\n"
            "      \"targets\": [\"disease A\", \"disease B\"],\n"
            "      \"reason\": \"why this matters\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Do not add narration."
        )
        if retry_hint:
            prompt += f"\nCorrection needed: {retry_hint}"
        return prompt

    def _collect_opposing_expectations(
        self, hypotheses: Sequence[DiseaseHypothesis]
    ) -> List[str]:
        pairs: List[str] = []
        for idx, left in enumerate(hypotheses):
            for right in hypotheses[idx + 1 :]:
                for sym_left in left.key_symptoms:
                    for sym_right in right.key_symptoms:
                        if sym_left.name.lower() != sym_right.name.lower():
                            continue
                        if sym_left.expected_state not in {"present", "absent"}:
                            continue
                        if sym_right.expected_state not in {"present", "absent"}:
                            continue
                        if sym_left.expected_state == sym_right.expected_state:
                            continue
                        if not (sym_left.is_unknown() or sym_right.is_unknown()):
                            continue
                        pairs.append(
                            f"{sym_left.name}: {left.name} expects {sym_left.expected_state}, {right.name} expects {sym_right.expected_state}."
                        )
        unique: List[str] = []
        seen = set()
        for item in pairs:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
            if len(unique) >= 5:
                break
        return unique

    def _build_action_summary(self, decision: PlannerDecision) -> Optional[str]:
        if decision.should_finish and decision.chosen_diagnosis:
            return f"Proceed to DIAGNOSIS READY for {decision.chosen_diagnosis}."
        actions: List[str] = []
        for clarification in decision.discriminators[:2]:
            focus = clarification.get("focus") or ""
            if not focus:
                continue
            action_type = clarification.get("action_type", "clarify")
            reason = clarification.get("reason") or ""
            targets = clarification.get("targets") or []
            if isinstance(targets, str):
                targets = [targets]
            targets = [t for t in targets if t]
            detail_parts: List[str] = []
            if reason:
                detail_parts.append(reason)
            if targets:
                detail_parts.append(f"targets {', '.join(targets)}")
            detail = "; ".join(detail_parts)
            if detail:
                actions.append(f"{action_type.capitalize()} {focus} ({detail})")
            else:
                actions.append(f"{action_type.capitalize()} {focus}")
        if actions:
            return "; ".join(actions)
        top = decision.top_hypothesis()
        if not top:
            return None
        unknowns = top.top_unknowns(limit=2)
        if unknowns:
            phrases = []
            for sym in unknowns:
                if sym.expected_state == "present":
                    phrases.append(f"confirm presence of {sym.name}")
                elif sym.expected_state == "absent":
                    phrases.append(f"confirm absence of {sym.name}")
                else:
                    phrases.append(f"clarify {sym.name}")
            return "Prioritize " + "; ".join(phrases)
        contradictions = top.contradictory_features(limit=2)
        if contradictions:
            names = ", ".join(sym.name for sym in contradictions)
            return f"Resolve contradictory evidence: {names}"
        return None

    def _maybe_restore_min_requirement(self) -> None:
        if self._min_symptoms_active < self.min_symptoms_requested:
            self._min_symptoms_active = min(
                self.min_symptoms_requested, self._min_symptoms_active + 1
            )
            if self._min_symptoms_active == self.min_symptoms_requested:
                self._reduction_note = None



def _normalize_presence(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    negatives = [
        "absent",
        "negative",
        "no",
        "denies",
        "without",
        "not present",
        "lacks",
        "lack",
        "ruled out",
        "none",
    ]
    for marker in negatives:
        if marker in text:
            return "absent"
    positives = [
        "present",
        "positive",
        "yes",
        "reports",
        "with",
        "observed",
        "found",
        "detected",
        "evidence of",
        "shows",
    ]
    for marker in positives:
        if marker in text:
            return "present"
    return "unknown"


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _safe_load_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass
    return None
