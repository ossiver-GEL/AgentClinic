import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from llm import query_model
from agents.agent_tools import load_prompts_json
from .test_graph import (
    canonicalize_test_name,
    cost_level,
    display_name,
    missing_prerequisites,
    resolve_test_name,
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


_TEMPLATE_PATTERN = re.compile(r"\{([a-zA-Z0-9_]+)\}")


def _render_template(template: str, mapping: Dict[str, Any]) -> str:
    def repl(match: 're.Match[str]') -> str:
        key = match.group(1)
        if key not in mapping:
            raise KeyError(key)
        value = mapping[key]
        return value if isinstance(value, str) else str(value)

    result = _TEMPLATE_PATTERN.sub(repl, template)
    if _TEMPLATE_PATTERN.search(result):
        leftovers = sorted({m.group(1) for m in _TEMPLATE_PATTERN.finditer(result)})
        raise KeyError(f"Unresolved template keys: {leftovers}")
    return result

_LOG_DIR = os.path.join(os.path.dirname(__file__), 'internal_runs')
_LOG_DIR_LOCK = threading.Lock()


def _ensure_log_dir() -> str:
    with _LOG_DIR_LOCK:
        if not os.path.isdir(_LOG_DIR):
            os.makedirs(_LOG_DIR, exist_ok=True)
    return _LOG_DIR


def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in obj]
    return repr(obj)


class EnhancedDoctorAgent:

    def __init__(self, scenario, backend_str="gpt-4o-mini", max_infs=20, bias_present=None, img_request=False) -> None:
        self.infs = 0
        self.MAX_INFS = max_infs
        self.agent_hist = ""
        self.presentation = ""
        self.backend = backend_str
        self.bias_present = (None if bias_present == "None" else bias_present)
        self.scenario = scenario
        self.pipe = None
        self.img_request = img_request
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
        self.prompts = load_prompts_json("enhanced_doctor")
        self.confidence_threshold = 0.9
        self.reassessment_floor = 0.05
        self.max_plan_items = 5
        self._log_path: Optional[str] = ''
        self._session_id = ''
        self._log_sequence = 0
        self._log_error_reported = False
        self.reset()

    def generate_bias(self) -> str:
        if self.bias_present is None:
            return ""
        prompts = self.prompts.get("biases", {})
        if self.bias_present in prompts:
            return prompts[self.bias_present]
        print(f"BIAS TYPE {self.bias_present} NOT SUPPORTED, ignoring bias...")
        return ""

    def _setup_logging(self) -> None:
        log_dir = _ensure_log_dir()
        self._session_id = uuid4().hex
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self._log_path = os.path.join(log_dir, f"run_{stamp}_{self._session_id[:8]}.jsonl")
        self._log_sequence = 0
        self._log_error_reported = False
        self._log_internal('log_session_opened', {'log_path': self._log_path})

    def _log_internal(self, event: str, payload: Dict[str, Any]) -> None:
        if not getattr(self, "_log_path", None):
            return
        record = {
            "session_id": self._session_id,
            "event": event,
            "turn_index": self.infs,
            "timestamp": time.time(),
            "payload": _make_json_safe(payload),
        }
        record["index"] = self._log_sequence
        self._log_sequence += 1
        try:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            if not self._log_error_reported:
                self._log_error_reported = True
                print(f"[EnhancedDoctorAgent] Logging failure: {exc}")

    def _confidence_snapshot(self) -> Dict[str, float]:
        return {name: round(float(info.get("confidence", 0.0)), 6) for name, info in self.diseases.items()}

    def inference_doctor(self, question: str, image_requested: bool = False) -> str:
        if self.infs >= self.MAX_INFS:
            self._log_internal("inference_limit_reached", {"question": question, "image_requested": image_requested})
            return "Maximum inferences reached"

        clean_question = (question or "").strip()
        self._log_internal("inference_turn_start", {"turn_index": self.infs, "question": clean_question, "image_requested": image_requested})

        if clean_question:
            self._handle_new_observation(clean_question)

        self._ensure_initial_assessment()
        self._recompute_confidences()

        if self._should_reassess():
            self._ensure_initial_assessment(force=True)
            self._recompute_confidences()

        if self._ready_to_diagnose():
            reply = self._prepare_diagnosis_reply()
            stage = "diagnosis"
        else:
            action = self._choose_next_action()
            reply = self._render_action(action)
            stage = "interaction"

        self.agent_hist += clean_question + "\n\n" + reply + "\n\n"
        self._log_internal("inference_turn_end", {"turn_index": self.infs, "stage": stage, "reply": reply, "confidences": self._confidence_snapshot(), "remaining_plan": self.priority_plan, "completed_tests": self.completed_tests, "pending_tests": self.pending_tests})
        self.infs += 1
        return reply

    def system_prompt(self) -> str:
        base = self.prompts["system_base"].format(self.MAX_INFS, self.infs)
        base_with_images = base + (self.prompts.get("system_images_suffix", "") if self.img_request else "")
        bias_prompt = self.generate_bias()
        presentation_suffix = self.prompts["system_presentation_suffix"].format(self.presentation)
        return base_with_images + bias_prompt + presentation_suffix

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()
        self.patient_profile = self.scenario.patient_information()
        exam_info = self.scenario.exam_information()
        self.physical_exam = {k: v for k, v in exam_info.items() if k != "tests"}
        tests_dict = exam_info.get("tests", {})
        if isinstance(tests_dict, dict):
            self.available_tests = list(tests_dict.keys())
        else:
            self.available_tests = tests_dict if isinstance(tests_dict, list) else []
        self.initial_assessment_done = False
        self.typical_features: List[Dict[str, Any]] = []
        self.diseases: Dict[str, Dict[str, Any]] = {}
        self.priority_plan: List[Dict[str, Any]] = []
        self.completed_tests: List[str] = []
        self.pending_tests: List[str] = []
        self.observation_log: List[Dict[str, Any]] = []
        self.last_action: Optional[Dict[str, Any]] = None
        self.plan_failures = 0

        self._setup_logging()
        scenario_meta = getattr(self.scenario, 'scenario_dict', None)
        scenario_id = None
        if isinstance(scenario_meta, dict):
            scenario_id = scenario_meta.get('Scenario_ID') or scenario_meta.get('scenario_id') or scenario_meta.get('id')
        self._log_internal(
            'reset',
            {
                'scenario_id': scenario_id,
                'presentation': self.presentation,
                'patient_profile': self.patient_profile,
                'physical_exam': self.physical_exam,
                'available_tests': self.available_tests,
            },
        )

    # ------------------------
    # Core orchestration
    # ------------------------
    def _ensure_initial_assessment(self, force: bool = False) -> None:
        if self.initial_assessment_done and not force:
            self._log_internal('ensure_initial_assessment_skip', {'force': force})
            return

        self._log_internal('ensure_initial_assessment_start', {'force': force})

        base_context = {
            "examiner_objective": self.presentation,
            "patient_profile": self.patient_profile,
            "physical_exam": self.physical_exam,
            "available_tests": self.available_tests[:20],
            "observation_summary": self._summarize_observations(limit=6),
        }
        self._log_internal('ensure_initial_assessment_context', base_context)

        previous_state = self.diseases if force else {}

        features_payload = self._call_json_prompt(
            "feature_extractor",
            {
                "basic_info": json.dumps(base_context, ensure_ascii=False),
            },
        )
        if not isinstance(features_payload, dict):
            raise ValueError("feature_extractor response must be a JSON object")
        self.typical_features = features_payload.get("typical_features", [])
        self._log_internal('ensure_initial_assessment_features', {'typical_features': self.typical_features})

        diseases_payload = self._call_json_prompt(
            "disease_builder",
            {
                "basic_info": json.dumps(base_context, ensure_ascii=False),
                "typical_features": json.dumps(self.typical_features, ensure_ascii=False),
                "existing_diseases": json.dumps(self._serialize_disease_table_for_prompt(include_reason=True), ensure_ascii=False),
            },
        )
        if not isinstance(diseases_payload, dict):
            raise ValueError("disease_builder response must be a JSON object")
        self._ingest_disease_table(diseases_payload, previous_state)
        self._log_internal('ensure_initial_assessment_diseases', {'diseases': self._serialize_disease_table_for_prompt(include_reason=True)})
        self.initial_assessment_done = True
        self.priority_plan = []
        self._log_internal('ensure_initial_assessment_complete', {'confidences': self._confidence_snapshot()})

    def _handle_new_observation(self, observation: str) -> None:
        observation_type = "patient_response"
        if self.last_action and self.last_action.get("type") == "test":
            observation_type = "measurement_result"
            last_test = self.last_action.get("test_name")
            if last_test:
                canonical, _ = resolve_test_name(last_test)
                if canonical in self.pending_tests:
                    self.pending_tests = [t for t in self.pending_tests if t != canonical]
                if canonical not in self.completed_tests:
                    self.completed_tests.append(canonical)
        self.observation_log.append({"type": observation_type, "text": observation})
        self._log_internal('observation_received', {'type': observation_type, 'text': observation, 'last_action': self.last_action})

        status_payload = self._call_json_prompt(
            "status_update",
            {
                "observation": json.dumps(
                    {
                        "latest": observation,
                        "type": observation_type,
                        "previous_action": self.last_action,
                        "recent_observations": self._summarize_observations(limit=6),
                    },
                    ensure_ascii=False,
                ),
                "disease_table": json.dumps(
                    self._serialize_disease_table_for_prompt(include_reason=True),
                    ensure_ascii=False,
                ),
            },
        )
        if not isinstance(status_payload, dict):
            raise ValueError("status_update response must be a JSON object")
        self._apply_status_updates(status_payload)
        if status_payload.get("reconsider_diseases"):
            self.initial_assessment_done = False
        self.priority_plan = []
        self._log_internal('status_update_applied', {'payload': status_payload, 'completed_tests': self.completed_tests, 'pending_tests': self.pending_tests, 'confidences': self._confidence_snapshot()})

    def _apply_status_updates(self, payload: Dict[str, Any]) -> None:
        updates = payload.get("updates", [])
        for update in updates:
            if not isinstance(update, dict):
                continue
            disease_name = update.get("disease")
            feature_name = update.get("feature")
            if not disease_name or not feature_name:
                continue
            disease = self.diseases.get(disease_name)
            if not disease:
                continue
            feature = disease["features"].get(feature_name)
            if not feature:
                continue
            if "status" in update:
                feature["status"] = int(_clamp(float(update["status"]), -100, 100))
            if "status_reason" in update:
                feature["status_reason"] = str(update["status_reason"])
            if "weight" in update:
                feature["weight"] = float(_clamp(float(update["weight"]), 0, 100))
            if "recommended_test" in update and update["recommended_test"]:
                feature["recommended_test"] = str(update["recommended_test"])
            if "conflicts_with" in update and isinstance(update["conflicts_with"], list):
                feature["conflicts_with"] = [str(c) for c in update["conflicts_with"]]
            if "cost_hint" in update:
                feature["cost_hint"] = str(update["cost_hint"])
        new_features = payload.get("new_features", [])
        for entry in new_features:
            if not isinstance(entry, dict):
                continue
            disease_name = entry.get("disease")
            feature_name = entry.get("feature")
            if not disease_name or not feature_name:
                continue
            disease = self.diseases.setdefault(
                disease_name,
                {"name": disease_name, "rationale": entry.get("rationale", ""), "features": {}, "confidence": 0.0},
            )
            feature = disease["features"].get(feature_name, {})
            feature["name"] = feature_name
            feature["weight"] = float(_clamp(float(entry.get("weight", 40.0)), 0, 100))
            feature["status"] = int(_clamp(float(entry.get("status", 0)), -100, 100))
            feature["status_reason"] = str(entry.get("status_reason", ""))
            feature["data_type"] = str(entry.get("data_type", "history"))
            feature["recommended_test"] = entry.get("recommended_test")
            feature["cost_hint"] = str(entry.get("cost_hint", "moderate"))
            feature["conflicts_with"] = [str(c) for c in entry.get("conflicts_with", [])]
            disease["features"][feature_name] = feature
        if payload.get("prune_diseases"):
            keep = set(str(name) for name in payload["prune_diseases"] if isinstance(name, str))
            if keep:
                self.diseases = {k: v for k, v in self.diseases.items() if k in keep}

    def _ingest_disease_table(self, payload: Dict[str, Any], previous: Dict[str, Dict[str, Any]]) -> None:
        diseases_list = payload.get("diseases", [])
        if not isinstance(diseases_list, list) or not diseases_list:
            raise ValueError("disease_builder must return a non-empty 'diseases' list")
        new_table: Dict[str, Dict[str, Any]] = {}
        for disease_entry in diseases_list:
            if not isinstance(disease_entry, dict):
                continue
            disease_name = disease_entry.get("name")
            if not disease_name:
                continue
            features_block = disease_entry.get("features", [])
            disease_state = {
                "name": disease_name,
                "rationale": disease_entry.get("rationale", ""),
                "features": {},
                "confidence": 0.0,
            }
            prev_features = {}
            if disease_name in previous:
                prev_features = previous[disease_name].get("features", {})
            for feature_entry in features_block:
                if not isinstance(feature_entry, dict):
                    continue
                feature_name = feature_entry.get("name")
                if not feature_name:
                    continue
                prev_feature = prev_features.get(feature_name, {})
                status_raw = feature_entry.get("status", 0)
                prev_status = prev_feature.get("status")
                status_val = int(_clamp(float(status_raw), -100, 100))
                if prev_status is not None and abs(prev_status) > abs(status_val):
                    status_val = int(_clamp(float(prev_status), -100, 100))
                weight_val = float(_clamp(float(feature_entry.get("weight", prev_feature.get("weight", 40))), 0, 100))
                feature_state = {
                    "name": feature_name,
                    "weight": weight_val,
                    "status": status_val,
                    "status_reason": feature_entry.get("status_reason", prev_feature.get("status_reason", "")),
                    "data_type": feature_entry.get("data_type", prev_feature.get("data_type", "history")),
                    "recommended_test": feature_entry.get("recommended_test", prev_feature.get("recommended_test")),
                    "cost_hint": feature_entry.get("cost_hint", prev_feature.get("cost_hint", "moderate")),
                    "conflicts_with": feature_entry.get("conflicts_with", prev_feature.get("conflicts_with", [])),
                }
                disease_state["features"][feature_name] = feature_state
            if disease_state["features"]:
                new_table[disease_name] = disease_state
        if not new_table:
            raise ValueError("No diseases extracted from disease_builder response")
        self.diseases = new_table

    def _recompute_confidences(self) -> None:
        for disease in self.diseases.values():
            features = list(disease["features"].values())
            total_weight = sum(max(float(f.get("weight", 0.0)), 0.0) for f in features)
            if total_weight <= 0:
                disease["confidence"] = 0.0
                continue
            known = [f for f in features if abs(float(f.get("status", 0))) > 0]
            coverage = len(known) / max(len(features), 1)
            weighted_sum = sum(max(float(f.get("weight", 0.0)), 0.0) * (float(f.get("status", 0)) / 100.0) for f in features)
            normalized = (weighted_sum + total_weight) / (2 * total_weight)
            normalized = _clamp(normalized, 0.0, 1.0)
            disease["confidence"] = normalized * coverage

        self._log_internal('confidences_recomputed', {'confidences': self._confidence_snapshot()})

    def _ready_to_diagnose(self) -> bool:
        if not self.diseases:
            return False
        top_conf = max(d["confidence"] for d in self.diseases.values())
        if top_conf >= self.confidence_threshold:
            return True
        if self.infs >= self.MAX_INFS - 1:
            return True
        return False

    def _prepare_diagnosis_reply(self) -> str:
        if not self.diseases:
            self._log_internal('diagnosis_ready', {'diagnosis': 'Unknown', 'confidences': {}})
            return "DIAGNOSIS READY: Unknown"
        top = max(self.diseases.values(), key=lambda d: d.get("confidence", 0.0))
        diagnosis = top.get("name", "Unknown")
        self._log_internal('diagnosis_ready', {'diagnosis': diagnosis, 'confidences': self._confidence_snapshot()})
        return f"DIAGNOSIS READY: {diagnosis}"

    def _choose_next_action(self) -> Dict[str, Any]:
        if not self.priority_plan:
            plan_payload = self._call_json_prompt(
                "priority_planner",
                {
                    "disease_table": json.dumps(
                        self._serialize_disease_table_for_prompt(include_reason=True),
                        ensure_ascii=False,
                    ),
                    "completed_tests": json.dumps(self.completed_tests, ensure_ascii=False),
                    "pending_tests": json.dumps(self.pending_tests, ensure_ascii=False),
                    "max_items": self.max_plan_items,
                },
            )
            decisions = plan_payload.get("decisions", []) if isinstance(plan_payload, dict) else []
            if not isinstance(decisions, list):
                raise ValueError("priority_planner must return decisions list")
            self.priority_plan = decisions
            self._log_internal('priority_plan_ready', {'decisions': self.priority_plan})
        if not self.priority_plan:
            fallback = self._fallback_question()
            self.last_action = fallback
            self._log_internal('action_selected', {'action': fallback, 'source': 'fallback'})
            return fallback
        action = self.priority_plan.pop(0)
        structured = self._normalize_action(action)
        self.last_action = structured
        self._log_internal('action_selected', {'action': structured, 'original_decision': action, 'source': 'plan', 'remaining_plan': self.priority_plan})
        return structured

    def _normalize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        act_type = action.get("action_type", "ask")
        feature = action.get("feature", "")
        justification = action.get("justification", "")
        if act_type == "test":
            requested_test = action.get("test_name") or feature
            resolved_name, _ = resolve_test_name(requested_test)
            cost_tag = cost_level(resolved_name)
            missing = missing_prerequisites(resolved_name, self.completed_tests, self.pending_tests)
            urgency = str(action.get("urgency", "normal")).lower()
            target_display = display_name(requested_test)
            if missing and urgency != "urgent":
                next_test = missing[0]
                prereq_cost = cost_level(next_test)
                pretty_name = display_name(next_test)
                canonical = canonicalize_test_name(next_test)
                if canonical not in self.pending_tests:
                    self.pending_tests.append(canonical)
                return {
                    "type": "test",
                    "feature": feature,
                    "reason": f"Stepwise approach: perform {pretty_name} ({prereq_cost} cost) before {target_display}",
                    "test_name": pretty_name,
                }
            pretty_name = target_display
            canonical = canonicalize_test_name(requested_test)
            if canonical not in self.pending_tests:
                self.pending_tests.append(canonical)
            target_label = feature or "the leading diagnosis"
            reason_text = justification or f"Need {pretty_name} ({cost_tag} cost) to clarify {target_label}"
            return {
                "type": "test",
                "feature": feature,
                "reason": reason_text,
                "test_name": pretty_name,
            }
        question = action.get("question") or self._default_question_for_feature(feature)
        question = self._ensure_question_format(question)
        return {
            "type": "question",
            "feature": feature,
            "reason": justification,
            "question": question,
        }

    def _default_question_for_feature(self, feature: str) -> str:
        if not feature:
            return "Can you describe more about your main symptoms in detail?"
        return f"Can you tell me whether you have noticed {feature.lower()}?"

    def _render_action(self, action: Dict[str, Any]) -> str:
        if action.get("type") == "test":
            test_name = action.get("test_name") or ""
            if not test_name:
                raise ValueError("Test action missing test_name")
            return f"REQUEST TEST: {test_name}"
        question = action.get("question")
        if not question:
            raise ValueError("Question action missing question text")
        return question
    def _fallback_question(self) -> Dict[str, Any]:
        feature = self._find_uncertain_feature()
        question = self._default_question_for_feature(feature)
        question = self._ensure_question_format(question)
        return {
            "type": "question",
            "feature": feature,
            "reason": "Fallback exploration",
            "question": question,
        }

    def _find_uncertain_feature(self) -> str:
        best_feature = ""
        best_weight = -1.0
        for disease in self.diseases.values():
            for feat in disease["features"].values():
                status = abs(float(feat.get("status", 0)))
                weight = float(feat.get("weight", 0.0))
                if status <= 20 and weight > best_weight:
                    best_weight = weight
                    best_feature = feat.get("name", "")
        return best_feature

    def _ensure_question_format(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "Could you tell me more about your symptoms?"
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) > 3:
            sentences = sentences[:3]
        text = " ".join(sentences)
        text = text.rstrip(".?!") + "?"
        return text

    def _serialize_disease_table_for_prompt(self, include_reason: bool = False, limit_features: int = 6) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for disease in self.diseases.values():
            features = list(disease["features"].values())
            features_sorted = sorted(features, key=lambda f: float(f.get("weight", 0.0)), reverse=True)
            formatted = []
            for feat in features_sorted[:limit_features]:
                formatted.append(
                    {
                        "name": feat.get("name"),
                        "weight": feat.get("weight"),
                        "status": feat.get("status"),
                        "status_reason": feat.get("status_reason"),
                        "data_type": feat.get("data_type"),
                        "recommended_test": feat.get("recommended_test"),
                        "cost_hint": feat.get("cost_hint"),
                        "conflicts_with": feat.get("conflicts_with", []),
                    }
                )
            entry = {
                "name": disease.get("name"),
                "confidence": disease.get("confidence", 0.0),
                "features": formatted,
            }
            if include_reason:
                entry["rationale"] = disease.get("rationale", "")
            summary.append(entry)
        return summary

    def _summarize_observations(self, limit: int = 5) -> List[Dict[str, Any]]:
        tail = self.observation_log[-limit:]
        summary = []
        offset = len(self.observation_log) - len(tail)
        for idx, obs in enumerate(tail):
            summary.append(
                {
                    "turn": offset + idx + 1,
                    "type": obs.get("type"),
                    "text": obs.get("text"),
                }
            )
        return summary

    def _call_json_prompt(self, section: str, template_args: Dict[str, Any], max_attempts: int = 3) -> Dict[str, Any]:
        analysis_prompts = self.prompts.get("analysis", {})
        if section not in analysis_prompts:
            raise KeyError(f"Missing prompt section analysis.{section}")
        section_prompts = analysis_prompts[section]
        system_prompt = section_prompts["system"]
        user_template = section_prompts["user_template"]
        payload = dict(template_args)
        payload.setdefault("error_hint", "")
        last_error = ""
        for attempt in range(max_attempts):
            attempt_index = attempt + 1
            user_prompt = _render_template(user_template, payload)
            payload_snapshot = dict(payload)
            self._log_internal("analysis_request", {"section": section, "attempt": attempt_index, "payload": payload_snapshot, "user_prompt": user_prompt})
            raw = query_model(self.backend, user_prompt, system_prompt, scene=self.scenario)
            try:
                json_text = self._extract_json(raw)
                parsed = json.loads(json_text)
                self._log_internal("analysis_response", {"section": section, "attempt": attempt_index, "raw": raw, "parsed": parsed})
                return parsed
            except Exception as exc:
                last_error = str(exc)
                self._log_internal("analysis_response_error", {"section": section, "attempt": attempt_index, "error": last_error, "raw": raw})
                payload["error_hint"] = (
                    "\nPrevious attempt failed: "
                    + last_error
                    + " Please re-issue the response strictly as JSON matching the requested schema."
                )
        self._log_internal("analysis_failure", {"section": section, "attempts": max_attempts, "last_error": last_error})
        raise RuntimeError(f"Failed to obtain valid JSON from section '{section}' after {max_attempts} attempts: {last_error}")


    @staticmethod
    def _extract_json(text: str) -> str:
        if not text:
            raise ValueError("Empty response")
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return stripped
        obj_start = stripped.find("{")
        obj_end = stripped.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            return stripped[obj_start : obj_end + 1]
        arr_start = stripped.find("[")
        arr_end = stripped.rfind("]")
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            return stripped[arr_start : arr_end + 1]
        raise ValueError("No JSON object or array found in response")

    def _should_reassess(self) -> bool:
        if not self.diseases:
            return False
        max_conf = max(disease.get("confidence", 0.0) for disease in self.diseases.values())
        known = sum(
            1
            for disease in self.diseases.values()
            for feature in disease["features"].values()
            if abs(float(feature.get("status", 0))) > 0
        )
        return max_conf <= self.reassessment_floor and known >= 3
