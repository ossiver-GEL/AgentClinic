import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from llm import query_model
from agents.agent_tools import load_prompts_json


@dataclass
class MethodInstance:
    name: str
    method_type: str
    importance: float
    diseases: Tuple[str, ...]
    expected: Dict[str, List[str]]
    instruction: Optional[str] = None
    execution: Optional[str] = None
    source: str = "disease"
    normalized_name: str = ""
    normalized_type: str = ""
    diff_weight: float = 0.0


@dataclass
class PlanMethod:
    name: str
    method_type: str
    diff_weight: float
    importance: float
    combined_priority: float
    instances: List[MethodInstance] = field(default_factory=list)
    instructions: List[str] = field(default_factory=list)
    execution_hints: List[str] = field(default_factory=list)
    related_diseases: List[str] = field(default_factory=list)
    status: str = "pending"


@dataclass
class DiseaseState:
    name: str
    rationale: str = ""
    active: bool = True
    method_importance: Dict[str, float] = field(default_factory=dict)
    method_expectations: Dict[str, List[str]] = field(default_factory=dict)
    method_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


class EnhancedDoctorAgent:
    def __init__(self, scenario, backend_str: str = "gpt-4o-mini", max_infs: int = 20, bias_present=None, img_request: bool = False) -> None:
        self.infs = 0
        self.MAX_INFS = max_infs
        self.agent_hist = ""
        self.backend = backend_str
        self.bias_present = (None if bias_present == "None" else bias_present)
        self.scenario = scenario
        self.pipe = None
        self.img_request = img_request
        self.prompts = load_prompts_json("enhanced_doctor")
        self.flow_prompts = self.prompts.get("flows", {})
        self.question_prompt = self.prompts.get("question_prompt", {})
        self.settings = self.prompts.get("settings", {})
        self.confirm_threshold = float(self.settings.get("confirm_threshold", 90.0))
        self.elimination_threshold = float(self.settings.get("elimination_threshold", -5.0))
        self.max_retries = int(self.settings.get("max_retries", 3))
        self.biases = [
            "recency",
            "frequency",
            "false_consensus",
            "confirmation",
            "status_quo",
            "gender",
            "race",
            "sexual_orientation",
            "cultural",
            "education",
            "religion",
            "socioeconomic",
        ]
        self.reset()

    def generate_bias(self) -> str:
        if self.bias_present is None:
            return ""
        prompts = self.prompts.get("biases", {})
        if self.bias_present in prompts:
            return prompts[self.bias_present]
        print(f"BIAS TYPE {self.bias_present} NOT SUPPORTED, ignoring bias...")
        return ""

    def inference_doctor(self, question: str, image_requested: bool = False) -> str:
        if self.infs >= self.MAX_INFS:
            return "Maximum inferences reached"
        incoming = (question or "").strip()
        if incoming:
            self._process_incoming_information(incoming)
            if self.final_diagnosis:
                output = self._prepare_diagnosis_output()
                self._record_turn(incoming, output)
                return output
        if not self.plan_initialized:
            self._ensure_initial_summary()
            self._initialize_plan()
        if self.final_diagnosis:
            output = self._prepare_diagnosis_output()
            self._record_turn(incoming, output)
            return output
        remaining_turns = self.MAX_INFS - self.infs
        if remaining_turns <= 1:
            output = self._diagnose_with_best_guess()
            self._record_turn(incoming, output)
            return output
        next_method = self._select_next_method()
        if next_method is None:
            output = self._diagnose_with_best_guess()
            self._record_turn(incoming, output)
            return output
        if next_method.method_type == "question":
            output = self._execute_question_method(next_method)
        else:
            output = self._execute_test_method(next_method)
        self._record_turn(incoming, output)
        return output

    def system_prompt(self) -> str:
        conversation = self.prompts.get("conversation", {})
        base = conversation.get("system_base", "").format(self.MAX_INFS, self.infs)
        suffix = conversation.get("system_images_suffix", "") if self.img_request else ""
        presentation = conversation.get("system_presentation_suffix", "{}").format(self.presentation)
        return base + suffix + (self.bias_prompt or "") + presentation

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self._format_presentation(self.scenario.examiner_information())
        self.bias_prompt = self.generate_bias()
        self.knowledge_summary = ""
        self.diseases: Dict[str, DiseaseState] = {}
        self.method_instances: List[MethodInstance] = []
        self.plan_methods: List[PlanMethod] = []
        self.completed_methods: set = set()
        self.awaiting_method: Optional[PlanMethod] = None
        self.awaiting_method_context: Optional[Dict[str, Any]] = None
        self.plan_initialized = False
        self.pair_processed: set = set()
        self.disease_methods_processed: set = set()
        self.final_diagnosis: Optional[str] = None
        self.diagnosis_delivered = False

    # --- Internal helpers ---
    def _record_turn(self, incoming: str, outgoing: str) -> None:
        self.agent_hist += incoming + "\n\n" + outgoing + "\n\n"
        self.infs += 1

    def _format_presentation(self, presentation: Any) -> str:
        if isinstance(presentation, (dict, list)):
            try:
                return json.dumps(presentation, ensure_ascii=False, indent=2)
            except TypeError:
                return str(presentation)
        return str(presentation)

    def _ensure_initial_summary(self) -> None:
        if self.knowledge_summary:
            return
        payload = {
            "previous_summary": "",
            "new_observation": {
                "type": "initial_presentation",
                "presentation": self.presentation,
                "bias_context": self.bias_prompt.strip() if self.bias_prompt else "",
            },
        }
        result = self._call_llm_json("knowledge_update", payload)
        summary = result.get("updated_summary")
        if not summary:
            raise ValueError("Knowledge update did not return summary.")
        self.knowledge_summary = summary

    def _initialize_plan(self) -> None:
        candidates = self._call_differential()
        new_diseases: List[str] = []
        for cand in candidates:
            name = self._require_str(cand.get("name"), "candidate.name")
            rationale = cand.get("rationale", "")
            if name not in self.diseases:
                self.diseases[name] = DiseaseState(name=name, rationale=rationale)
                new_diseases.append(name)
            else:
                state = self.diseases[name]
                state.active = True
                if rationale:
                    state.rationale = rationale
        if not new_diseases:
            new_diseases = [name for name, state in self.diseases.items() if state.active]
        if not new_diseases:
            raise ValueError("Differential generation returned no active diseases.")
        self._collect_new_disease_methods(new_diseases)
        self._normalize_and_build_plan()
        self.plan_initialized = True

    def _select_next_method(self) -> Optional[PlanMethod]:
        pending = [pm for pm in self.plan_methods if pm.status == "pending"]
        if not pending:
            return None
        pending.sort(key=lambda item: (-item.combined_priority, item.name))
        method = pending[0]
        method.status = "awaiting_result"
        self.awaiting_method = method
        return method

    def _execute_question_method(self, plan_method: PlanMethod) -> str:
        payload = {
            "method_name": plan_method.name,
            "known_information_summary": self.knowledge_summary,
            "target_instructions": self._unique(plan_method.instructions) or [plan_method.name],
            "bias_context": self.bias_prompt.strip() if self.bias_prompt else "",
            "asked_turns": self.infs,
            "remaining_turns": max(self.MAX_INFS - self.infs, 0),
        }
        if not self.question_prompt:
            raise ValueError("Question prompt configuration missing.")
        system_prompt = self.question_prompt.get("system", "")
        user_header = self.question_prompt.get("user_header", "")
        parsed: Optional[Dict[str, Any]] = None
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            user_prompt = self._build_user_prompt_from_header(user_header, payload, attempt)
            raw = self._invoke_with_retry(system_prompt, user_prompt)
            try:
                parsed = json.loads(raw)
                break
            except json.JSONDecodeError as err:
                last_error = err
        if parsed is None:
            raise ValueError(f"Question generation returned non-JSON after {self.max_retries} attempts: {last_error}")
        question_text = self._require_str(parsed.get("question"), "question")
        plan_method.status = "awaiting_result"
        self.awaiting_method_context = {
            "method_name": plan_method.name,
            "method_type": plan_method.method_type,
            "question": question_text,
        }
        return question_text.strip()

    def _execute_test_method(self, plan_method: PlanMethod) -> str:
        test_name = self._unique(plan_method.execution_hints) or [plan_method.name]
        command = test_name[0]
        plan_method.status = "awaiting_result"
        self.awaiting_method_context = {
            "method_name": plan_method.name,
            "method_type": plan_method.method_type,
            "test_request": command,
        }
        return f"REQUEST TEST: {command}"

    def _process_incoming_information(self, observation: str) -> None:
        if self.awaiting_method is None:
            context = {
                "method_name": None,
                "observation": observation,
            }
            self._apply_knowledge_update(context)
            self._refresh_differential()
            return
        method = self.awaiting_method
        context = dict(self.awaiting_method_context or {})
        context["observation"] = observation
        self.awaiting_method_context = None
        self.awaiting_method = None
        self.completed_methods.add(method.name)
        method.status = "completed"
        self._score_method_result(method, observation)
        self._apply_knowledge_update(context)
        self._refresh_differential()

    def _score_method_result(self, method: PlanMethod, observation: str) -> None:
        expectation_map: Dict[str, List[str]] = {}
        weight_map: Dict[str, float] = {}
        for disease in method.related_diseases:
            state = self.diseases.get(disease)
            if not state or not state.active:
                continue
            expectations = state.method_expectations.get(method.name, [])
            if expectations:
                expectation_map[disease] = expectations
            weight_map[disease] = float(state.method_importance.get(method.name, 0.0))
        payload = {
            "method_name": method.name,
            "method_type": method.method_type,
            "observation": observation,
            "expectations": expectation_map,
            "importance_weights": weight_map,
            "known_information_summary": self.knowledge_summary,
        }
        response = self._call_llm_json("alignment_scoring", payload)
        alignments = response.get("disease_alignment", [])
        if not isinstance(alignments, list):
            raise ValueError("Alignment scoring returned invalid structure.")
        for item in alignments:
            disease = self._require_str(item.get("disease"), "alignment.disease")
            if disease not in self.diseases:
                continue
            alignment_value = self._read_float(item.get("alignment"), "alignment.score")
            state = self.diseases[disease]
            state.method_scores[method.name] = alignment_value
            self._recompute_confidence(state)
            if state.confidence >= self.confirm_threshold:
                self.final_diagnosis = disease
            elif state.confidence <= self.elimination_threshold:
                state.active = False

    def _apply_knowledge_update(self, context: Dict[str, Any]) -> None:
        payload = {
            "previous_summary": self.knowledge_summary,
            "new_observation": context,
            "bias_context": self.bias_prompt.strip() if self.bias_prompt else "",
        }
        result = self._call_llm_json("knowledge_update", payload)
        summary = result.get("updated_summary")
        if not summary:
            raise ValueError("Knowledge update did not return summary.")
        self.knowledge_summary = summary

    def _refresh_differential(self) -> None:
        candidates = self._call_differential()
        active_names = set()
        new_diseases: List[str] = []
        for cand in candidates:
            name = self._require_str(cand.get("name"), "candidate.name")
            rationale = cand.get("rationale", "")
            active_names.add(name)
            if name not in self.diseases:
                self.diseases[name] = DiseaseState(name=name, rationale=rationale)
                new_diseases.append(name)
            else:
                state = self.diseases[name]
                state.active = True
                if rationale:
                    state.rationale = rationale
        for name, state in list(self.diseases.items()):
            if name not in active_names and state.active and state.confidence <= self.elimination_threshold:
                state.active = False
        if new_diseases:
            self._collect_new_disease_methods(new_diseases)
        self._prune_inactive_diseases()
        self._normalize_and_build_plan()

    def _collect_new_disease_methods(self, diseases: List[str]) -> None:
        for disease in diseases:
            if disease in self.disease_methods_processed:
                continue
            payload = {
                "disease_name": disease,
                "known_information_summary": self.knowledge_summary,
                "bias_context": self.bias_prompt.strip() if self.bias_prompt else "",
                "max_methods": int(self.settings.get("max_methods_per_disease", 5)),
            }
            response = self._call_llm_json("disease_methods", payload)
            methods = response.get("methods", [])
            if not isinstance(methods, list) or not methods:
                raise ValueError(f"No methods returned for disease {disease}.")
            for method in methods:
                self._register_method_instance(method, (disease,), "disease")
            self.disease_methods_processed.add(disease)
        active_names = self._active_disease_names()
        for disease in diseases:
            for other in active_names:
                if other == disease:
                    continue
                pair = tuple(sorted((disease, other)))
                if pair in self.pair_processed:
                    continue
                payload = {
                    "disease_a": pair[0],
                    "disease_b": pair[1],
                    "known_information_summary": self.knowledge_summary,
                    "bias_context": self.bias_prompt.strip() if self.bias_prompt else "",
                    "max_methods": int(self.settings.get("max_methods_per_pair", 5)),
                }
                response = self._call_llm_json("pair_methods", payload)
                methods = response.get("methods", [])
                if not isinstance(methods, list) or not methods:
                    raise ValueError(f"No pairwise methods returned for {pair[0]} vs {pair[1]}.")
                for method in methods:
                    self._register_method_instance(method, pair, "pair")
                self.pair_processed.add(pair)

    def _register_method_instance(self, method: Dict[str, Any], diseases: Tuple[str, ...], source: str) -> None:
        name = self._require_str(method.get("name"), "method.name")
        method_type = self._require_str(method.get("method_type"), "method.method_type").lower()
        if method_type not in ("question", "test"):
            raise ValueError(f"Invalid method type '{method_type}' for {name}.")
        importance = self._read_float(method.get("importance"), "method.importance")
        instruction = method.get("instruction") or method.get("target")
        execution = method.get("test_name") or method.get("execution")
        expected: Dict[str, List[str]] = {}
        if source == "pair":
            outcomes = method.get("outcomes", [])
            if not isinstance(outcomes, list) or not outcomes:
                raise ValueError(f"Pair method {name} missing outcomes.")
            for item in outcomes:
                disease = self._require_str(item.get("supports"), "outcome.supports")
                detail = self._require_str(item.get("description") or item.get("outcome"), "outcome.description")
                expected.setdefault(disease, []).append(detail)
        else:
            findings = method.get("expected_findings")
            if isinstance(findings, list):
                expected[diseases[0]] = [self._require_str(f, "expected_findings[]") for f in findings]
            elif isinstance(findings, str):
                expected[diseases[0]] = [findings]
            else:
                raise ValueError(f"Disease method {name} missing expected findings.")
        instance = MethodInstance(
            name=name,
            method_type=method_type,
            importance=importance,
            diseases=diseases,
            expected=expected,
            instruction=instruction,
            execution=execution,
            source=source,
        )
        self.method_instances.append(instance)

    def _normalize_and_build_plan(self) -> None:
        active_names = set(self._active_disease_names())
        active_instances = [inst for inst in self.method_instances if all(d in active_names for d in inst.diseases)]
        if not active_instances:
            self.plan_methods = []
            for state in self.diseases.values():
                state.method_importance.clear()
                state.method_expectations.clear()
            return
        payload = {
            "methods": [
                {
                    "name": inst.name,
                    "method_type": inst.method_type,
                    "source": inst.source,
                    "instruction": inst.instruction or "",
                }
                for inst in active_instances
            ]
        }
        response = self._call_llm_json("method_normalization", payload)
        canon = response.get("canonical_methods", [])
        if not isinstance(canon, list) or not canon:
            raise ValueError("Method normalization returned no canonical methods.")
        name_map: Dict[str, str] = {}
        type_map: Dict[str, str] = {}
        for item in canon:
            normalized = self._require_str(item.get("normalized_name"), "canonical.normalized_name")
            method_type = self._require_str(item.get("method_type"), "canonical.method_type").lower()
            if method_type not in ("question", "test"):
                raise ValueError(f"Invalid canonical type '{method_type}' for {normalized}.")
            originals = item.get("original_names", [])
            if not isinstance(originals, list) or not originals:
                raise ValueError(f"Canonical method {normalized} missing original names.")
            for original in originals:
                name_map[self._require_str(original, "canonical.original_name")] = normalized
            type_map[normalized] = method_type
        for inst in active_instances:
            if inst.name not in name_map:
                raise ValueError(f"Canonical mapping missing for method {inst.name}.")
            inst.normalized_name = name_map[inst.name]
            inst.normalized_type = type_map[inst.normalized_name]
            inst.diff_weight = 2.0 if inst.source == "pair" else 1.0
        self._sync_disease_metadata(active_instances)
        plan_map: Dict[str, PlanMethod] = {}
        for inst in active_instances:
            key = inst.normalized_name
            if key not in plan_map:
                plan_map[key] = PlanMethod(
                    name=key,
                    method_type=inst.normalized_type,
                    diff_weight=0.0,
                    importance=0.0,
                    combined_priority=0.0,
                )
            plan = plan_map[key]
            plan.instances.append(inst)
            plan.diff_weight += inst.diff_weight
            if inst.importance > plan.importance:
                plan.importance = inst.importance
            if inst.instruction:
                plan.instructions.append(inst.instruction)
            if inst.execution:
                plan.execution_hints.append(inst.execution)
            for disease in inst.diseases:
                if disease not in plan.related_diseases:
                    plan.related_diseases.append(disease)
        for plan in plan_map.values():
            plan.instructions = self._unique(plan.instructions)
            plan.execution_hints = self._unique(plan.execution_hints)
            plan.combined_priority = plan.importance * plan.diff_weight
            if plan.name in self.completed_methods:
                plan.status = "completed"
            elif self.awaiting_method and plan.name == self.awaiting_method.name:
                plan.status = self.awaiting_method.status
            else:
                plan.status = "pending"
        self.plan_methods = list(plan_map.values())

    def _sync_disease_metadata(self, instances: List[MethodInstance]) -> None:
        active_names = set(self._active_disease_names())
        for state in self.diseases.values():
            if state.name not in active_names:
                state.method_importance.clear()
                state.method_expectations.clear()
                continue
            state.method_importance = {}
            state.method_expectations = {}
        for inst in instances:
            for disease in inst.diseases:
                state = self.diseases.get(disease)
                if not state or disease not in active_names:
                    continue
                current = state.method_importance.get(inst.normalized_name, 0.0)
                state.method_importance[inst.normalized_name] = current + inst.importance
                expectations = inst.expected.get(disease, [])
                if expectations:
                    bucket = state.method_expectations.setdefault(inst.normalized_name, [])
                    for note in expectations:
                        if note and note not in bucket:
                            bucket.append(note)
        for state in self.diseases.values():
            if state.name not in active_names:
                continue
            state.method_scores = {
                name: state.method_scores.get(name, 0.0)
                for name in state.method_importance.keys()
            }
            self._recompute_confidence(state)

    def _prune_inactive_diseases(self) -> None:
        active = set(self._active_disease_names())
        self.method_instances = [inst for inst in self.method_instances if all(d in active for d in inst.diseases)]
        self.pair_processed = {pair for pair in self.pair_processed if pair[0] in active and pair[1] in active}
        self.disease_methods_processed = {name for name in self.disease_methods_processed if name in active}

    def _call_differential(self) -> List[Dict[str, Any]]:
        payload = {
            "known_information_summary": self.knowledge_summary,
            "completed_methods": list(self.completed_methods),
            "active_diseases": self._active_disease_names(),
            "bias_context": self.bias_prompt.strip() if self.bias_prompt else "",
            "max_candidates": int(self.settings.get("max_candidates", 5)),
        }
        response = self._call_llm_json("differential", payload)
        candidates = response.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Differential generation returned no candidates.")
        return candidates

    def _call_llm_json(self, flow_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if flow_key not in self.flow_prompts:
            raise KeyError(f"Missing LLM flow configuration for '{flow_key}'.")
        flow = self.flow_prompts[flow_key]
        system_prompt = flow.get("system", "")
        user_header = flow.get("user_header", "")
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            user_prompt = self._build_user_prompt_from_header(user_header, payload, attempt)
            raw = self._invoke_with_retry(system_prompt, user_prompt)
            try:
                return json.loads(raw)
            except json.JSONDecodeError as err:
                last_error = err
        raise ValueError(f"LLM response for '{flow_key}' not valid JSON after {self.max_retries} attempts: {last_error}")

    def _invoke_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        return query_model(self.backend, user_prompt, system_prompt, scene=self.scenario)

    def _build_user_prompt_from_header(self, header: str, payload: Dict[str, Any], attempt: int) -> str:
        suffix = ""
        if attempt > 0:
            suffix = "\n\n# Previous response was invalid. Return ONLY valid JSON matching the schema."
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        header = header.strip()
        return f"{header}\n\nPAYLOAD:\n{payload_json}{suffix}"

    def _recompute_confidence(self, state: DiseaseState) -> None:
        if not state.method_importance:
            state.confidence = state.confidence if state.confidence else 0.0
            return
        numerator = 0.0
        denominator = 0.0
        for method_name, importance in state.method_importance.items():
            alignment = state.method_scores.get(method_name, 0.0)
            numerator += importance * alignment
            denominator += abs(importance) * 100.0
        if denominator == 0:
            state.confidence = 0.0
        else:
            state.confidence = (numerator / denominator) * 100.0

    def _diagnose_with_best_guess(self) -> str:
        if not self.diseases:
            raise ValueError("No diseases available for diagnosis.")
        best_state = max(self.diseases.values(), key=lambda ds: ds.confidence)
        self.final_diagnosis = best_state.name
        return self._prepare_diagnosis_output()

    def _prepare_diagnosis_output(self) -> str:
        if not self.final_diagnosis:
            return self._diagnose_with_best_guess()
        self.diagnosis_delivered = True
        return f"DIAGNOSIS READY: {self.final_diagnosis}"

    def _active_disease_names(self) -> List[str]:
        return [name for name, state in self.diseases.items() if state.active]

    def _unique(self, values: List[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for val in values:
            if not val or val in seen:
                continue
            seen.add(val)
            ordered.append(val)
        return ordered

    def _require_str(self, value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Field {field} missing or empty.")
        return value.strip()

    def _read_float(self, value: Any, field: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Field {field} expects float, got {value}.") from err
