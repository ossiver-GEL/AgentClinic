import json
import re
from typing import Any, Dict, List, Optional, Tuple

from llm import query_model
from agents.agent_tools import load_prompts_json


class EnhancedDoctorAgent:
    def __init__(
        self,
        scenario,
        backend_str: str = "gpt-4o-mini",
        max_infs: int = 20,
        bias_present: Optional[str] = None,
        img_request: bool = False,
    ) -> None:
        self.infs = 0
        self.MAX_INFS = max_infs
        self.agent_hist = ""
        self.presentation = ""
        self.backend = backend_str
        self.bias_present = None if bias_present == "None" else bias_present
        self.scenario = scenario
        self.pipe = None
        self.img_request = img_request

        self.prompts = load_prompts_json("enhanced_doctor")
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
        self.bias_prompt = self._resolve_bias()
        self.diagnosis_threshold = float(self.prompts.get("diagnosis_threshold", 0.9))
        self.reconsider_threshold = float(self.prompts.get("reconsider_threshold", 0.05))
        self.max_priority_items = int(self.prompts.get("max_priority_items", 5))

        # Test transition graph encodes manual progression from basic to advanced diagnostics
        self.test_graph: Dict[str, List[str]] = {
            "vitals": [],
            "bedside_screening": ["vitals"],
            "basic_lab": ["vitals", "bedside_screening"],
            "advanced_lab": ["basic_lab"],
            "basic_imaging": ["vitals", "bedside_screening"],
            "advanced_imaging": ["basic_imaging", "basic_lab"],
            "specialized": ["advanced_lab", "advanced_imaging"],
        }

        self.test_catalog: Dict[str, Dict[str, str]] = {
            "blood pressure": {"category": "vitals", "display": "Blood Pressure"},
            "pulse oximetry": {"category": "vitals", "display": "Pulse Oximetry"},
            "cbc": {"category": "basic_lab", "display": "Complete Blood Count"},
            "complete blood count": {"category": "basic_lab", "display": "Complete Blood Count"},
            "cmp": {"category": "basic_lab", "display": "Comprehensive Metabolic Panel"},
            "comprehensive metabolic panel": {"category": "basic_lab", "display": "Comprehensive Metabolic Panel"},
            "bmp": {"category": "basic_lab", "display": "Basic Metabolic Panel"},
            "basic metabolic panel": {"category": "basic_lab", "display": "Basic Metabolic Panel"},
            "urinalysis": {"category": "basic_lab", "display": "Urinalysis"},
            "hba1c": {"category": "basic_lab", "display": "HbA1c"},
            "d-dimer": {"category": "advanced_lab", "display": "D-Dimer"},
            "troponin": {"category": "advanced_lab", "display": "Troponin"},
            "arterial blood gas": {"category": "advanced_lab", "display": "Arterial Blood Gas"},
            "abg": {"category": "advanced_lab", "display": "Arterial Blood Gas"},
            "thyroid panel": {"category": "advanced_lab", "display": "Thyroid Function Panel"},
            "liver function tests": {"category": "basic_lab", "display": "Liver Function Tests"},
            "ultrasound abdomen": {"category": "basic_imaging", "display": "Ultrasound Abdomen"},
            "ultrasound": {"category": "basic_imaging", "display": "Ultrasound"},
            "chest x ray": {"category": "basic_imaging", "display": "Chest X-Ray"},
            "chest x-ray": {"category": "basic_imaging", "display": "Chest X-Ray"},
            "chest radiograph": {"category": "basic_imaging", "display": "Chest X-Ray"},
            "ecg": {"category": "bedside_screening", "display": "Electrocardiogram"},
            "ekg": {"category": "bedside_screening", "display": "Electrocardiogram"},
            "electrocardiogram": {"category": "bedside_screening", "display": "Electrocardiogram"},
            "spirometry": {"category": "bedside_screening", "display": "Spirometry"},
            "pulmonary function test": {"category": "advanced_lab", "display": "Pulmonary Function Test"},
            "echocardiogram": {"category": "advanced_imaging", "display": "Echocardiogram"},
            "stress test": {"category": "advanced_imaging", "display": "Cardiac Stress Test"},
            "ct chest": {"category": "advanced_imaging", "display": "CT Chest"},
            "computed tomography chest": {"category": "advanced_imaging", "display": "CT Chest"},
            "ct pulmonary angiography": {"category": "advanced_imaging", "display": "CT Pulmonary Angiography"},
            "cta": {"category": "advanced_imaging", "display": "CT Angiography"},
            "mri brain": {"category": "advanced_imaging", "display": "MRI Brain"},
            "brain mri": {"category": "advanced_imaging", "display": "MRI Brain"},
            "carotid ultrasound": {"category": "advanced_imaging", "display": "Carotid Ultrasound"},
            "coronary angiography": {"category": "specialized", "display": "Coronary Angiography"},
            "cardiac catheterization": {"category": "specialized", "display": "Cardiac Catheterization"},
            "colonoscopy": {"category": "specialized", "display": "Colonoscopy"},
            "endoscopy": {"category": "specialized", "display": "Upper Endoscopy"},
        }

        self.category_defaults: Dict[str, str] = {
            "vitals": "Blood Pressure",
            "bedside_screening": "Electrocardiogram",
            "basic_lab": "Complete Blood Count",
            "advanced_lab": "D-Dimer",
            "basic_imaging": "Chest X-Ray",
            "advanced_imaging": "CT Chest",
            "specialized": "Coronary Angiography",
        }

        self.reset()

    def _resolve_bias(self) -> str:
        if self.bias_present is None:
            return ""
        bias_bank = self.prompts.get("biases", {})
        if self.bias_present in bias_bank:
            return bias_bank[self.bias_present]
        print(f"BIAS TYPE {self.bias_present} NOT SUPPORTED, ignoring bias...")
        return ""

    def system_prompt(self) -> str:
        base = self.prompts.get("system_base", "")
        base_with_images = base + (
            self.prompts.get("system_images_suffix", "") if self.img_request else ""
        )
        presentation_tpl = self.prompts.get("system_presentation_suffix", "{}")
        presentation = presentation_tpl.format(self.presentation)
        bias = self.bias_prompt or ""
        return base_with_images + bias + presentation

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()
        self.observations: List[Dict[str, Any]] = []
        self.differential: Optional[Dict[str, Any]] = None
        self.feature_index: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        self.priority_plan: Optional[Dict[str, Any]] = None
        self.asked_features: set[str] = set()
        self.tests_requested: set[str] = set()
        self.completed_categories: set[str] = {"vitals"}
        self.initial_context = self._build_initial_context()

    def inference_doctor(self, question: str, image_requested: bool = False) -> str:
        if self.infs >= self.MAX_INFS:
            return "Maximum inferences reached"

        patient_message = (question or "").strip()
        if patient_message:
            self._record_observation(patient_message)

        self._ensure_differential(force=False)

        if patient_message:
            self._update_differential(patient_message)

        if self._needs_reassessment():
            self._ensure_differential(force=True)

        self.priority_plan = self._plan_next_steps()
        diagnosis_check = self._evaluate_diagnosis()

        final_response: Optional[str] = None

        if diagnosis_check.get("ready") or self.infs == self.MAX_INFS - 1:
            diagnosis_name = self._resolve_diagnosis_name(diagnosis_check)
            final_response = f"DIAGNOSIS READY: {diagnosis_name}"
        else:
            action_item = self._choose_action()
            if action_item is None:
                diagnosis_name = self._fallback_diagnosis_name()
                final_response = f"DIAGNOSIS READY: {diagnosis_name}"
            else:
                action = action_item.get("recommended_action", {})
                final_response = self._render_action(action, action_item)

        if final_response is None:
            raise RuntimeError("Failed to generate doctor response")

        self._append_history(patient_message, final_response)
        self.infs += 1

        if final_response.startswith("REQUEST TEST"):
            test_name = self._extract_test_from_utterance(final_response)
            if test_name:
                self._register_test(test_name)
        elif "REQUEST TEST" in final_response:
            test_name = self._extract_test_from_utterance(final_response)
            if test_name:
                self._register_test(test_name)

        return final_response

    def _build_initial_context(self) -> Dict[str, Any]:
        try:
            examiner = self.scenario.examiner_information()
        except AttributeError:
            examiner = {}
        try:
            patient = self.scenario.patient_information()
        except AttributeError:
            patient = {}
        try:
            baseline = self.scenario.exam_information()
        except AttributeError:
            baseline = {}
        return {
            "examiner_objective": examiner,
            "patient_background": patient,
            "baseline_findings": baseline,
        }

    def _record_observation(self, content: str) -> None:
        snapshot = content if len(content) <= 500 else content[:497] + "..."
        self.observations.append({"source": "patient", "content": snapshot})

    def _recent_observations(self, limit: int = 6) -> List[Dict[str, Any]]:
        return self.observations[-limit:]

    def _ensure_differential(self, force: bool) -> None:
        if self.differential is not None and not force:
            return
        payload = {
            "context": self.initial_context,
            "known_observations": self._recent_observations(),
            "requested_tests": sorted(self.tests_requested),
        }
        response = self._call_section("initial_differential", payload)
        data = self._json_loads(response)
        if "diseases" not in data:
            raise ValueError("Initial differential must contain 'diseases'")
        self.differential = data
        self._index_features()

    def _update_differential(self, observation: str) -> None:
        if not self.differential:
            return
        payload = {
            "current_assessment": self.differential,
            "new_observation": observation,
            "observation_window": self._recent_observations(),
            "asked_features": sorted(self.asked_features),
            "requested_tests": sorted(self.tests_requested),
        }
        response = self._call_section("differential_refinement", payload)
        data = self._json_loads(response)
        if "diseases" not in data:
            raise ValueError("Refined differential missing 'diseases'")
        self.differential = data
        self._index_features()

    def _needs_reassessment(self) -> bool:
        if not self.differential:
            return True
        diseases = self.differential.get("diseases", [])
        if not diseases:
            return True
        top_conf = max((d.get("confidence", 0.0) for d in diseases), default=0.0)
        if top_conf <= self.reconsider_threshold:
            return True
        # Check that there remain informative unresolved features to investigate
        unresolved = False
        for disease in diseases:
            for feat in disease.get("features", []):
                status = feat.get("status", 0)
                weight = feat.get("weight", 0)
                if abs(status) < 15 and weight >= 15:
                    unresolved = True
                    break
            if unresolved:
                break
        return not unresolved

    def _plan_next_steps(self) -> Dict[str, Any]:
        if not self.differential:
            raise ValueError("Cannot plan without differential")
        payload = {
            "assessment": self.differential,
            "asked_features": sorted(self.asked_features),
            "requested_tests": sorted(self.tests_requested),
            "completed_categories": sorted(self.completed_categories),
            "observation_summary": self._recent_observations(limit=4),
            "max_items": self.max_priority_items,
        }
        response = self._call_section("priority_planner", payload)
        data = self._json_loads(response)
        if "pending_features" not in data:
            raise ValueError("Priority planner must return 'pending_features'")
        return data

    def _evaluate_diagnosis(self) -> Dict[str, Any]:
        if not self.differential:
            return {"ready": False}
        payload = {
            "assessment": self.differential,
            "max_inferences": self.MAX_INFS,
            "questions_used": self.infs,
            "threshold": self.diagnosis_threshold,
        }
        response = self._call_section("diagnosis_evaluator", payload)
        data = self._json_loads(response)
        return data

    def _choose_action(self) -> Optional[Dict[str, Any]]:
        if not self.priority_plan:
            return None
        for item in self.priority_plan.get("pending_features", []):
            if not isinstance(item, dict):
                continue
            feature_id = item.get("feature_id")
            action = item.get("recommended_action") or {}
            action_type = (action.get("type") or "").lower()
            if feature_id and action_type != "request_test" and feature_id in self.asked_features:
                continue
            if action_type == "request_test":
                adjusted = self._apply_test_transition(action)
                item["recommended_action"] = adjusted
                action = adjusted
            if feature_id:
                self.asked_features.add(feature_id)
            if action:
                return item
        return None

    def _render_action(self, action: Dict[str, Any], plan_item: Dict[str, Any]) -> str:
        action_type = (action.get("type") or "").lower()
        if action_type == "request_test":
            test_name = action.get("test_name") or self._extract_test_from_utterance(action.get("utterance", ""))
            display = self._resolve_test_display(test_name)
            reason = action.get("rationale") or plan_item.get("rationale")
            reason = (reason or "I would like to gather more data").strip()
            if len(reason) > 120:
                reason = reason[:117].rstrip() + "..."
            return self._format_test_request(display, reason)
        prompt = action.get("utterance") or action.get("prompt") or action.get("content")
        if prompt:
            prompt = prompt.strip()
        if not prompt:
            feature_desc = plan_item.get("feature_description") or plan_item.get("description")
            if feature_desc:
                prompt = f"Could you clarify {feature_desc}?"
            else:
                prompt = "Could you tell me more about how you are feeling?"
        if not prompt.endswith("?") and "REQUEST TEST:" not in prompt and not prompt.startswith("DIAGNOSIS READY"):
            prompt = prompt.rstrip(".") + "?"
        sentences = re.split(r"(?<=[.!?])\s+", prompt.strip())
        if len(sentences) > 3:
            prompt = " ".join(sentences[:3])
        return prompt

    def _resolve_diagnosis_name(self, diag_payload: Dict[str, Any]) -> str:
        top = diag_payload.get("top_diagnosis", {}) if isinstance(diag_payload, dict) else {}
        name = top.get("name") or self._fallback_diagnosis_name()
        return name

    def _fallback_diagnosis_name(self) -> str:
        if not self.differential:
            return "Undifferentiated condition"
        diseases = self.differential.get("diseases", [])
        if not diseases:
            return "Undifferentiated condition"
        ordered = sorted(diseases, key=lambda d: d.get("confidence", 0.0), reverse=True)
        return ordered[0].get("name", "Undifferentiated condition")

    def _append_history(self, patient_message: str, doctor_reply: str) -> None:
        entry_parts: List[str] = []
        if patient_message:
            entry_parts.append(patient_message)
        entry_parts.append(doctor_reply)
        self.agent_hist += "\n\n".join(entry_parts) + "\n\n"

    def _index_features(self) -> None:
        self.feature_index.clear()
        if not self.differential:
            return
        for disease in self.differential.get("diseases", []):
            disease_name = disease.get("name", "")
            for feature in disease.get("features", []):
                desc = feature.get("description", "")
                fid = feature.get("id") or f"{disease_name}:{desc}".strip()
                feature["id"] = fid
                self.feature_index[fid] = (disease_name, feature)

    def _apply_test_transition(self, action: Dict[str, Any]) -> Dict[str, Any]:
        test_name = action.get("test_name") or self._extract_test_from_utterance(action.get("utterance", ""))
        if not test_name:
            return action
        category, display = self._resolve_test_category(test_name)
        if not category:
            action["test_name"] = display
            return action
        action["test_name"] = display
        prerequisites = self.test_graph.get(category, [])
        missing = [cat for cat in prerequisites if cat not in self.completed_categories]
        if not missing:
            return action
        urgency = (action.get("urgency") or "normal").lower()
        if urgency == "urgent":
            return action
        fallback_category = next((cat for cat in missing if cat in self.category_defaults), None)
        if fallback_category is None:
            return action
        fallback_test = self.category_defaults[fallback_category]
        normalized = self._normalize_test_name(fallback_test)
        if normalized in self.tests_requested:
            return action
        reason = f"Before we proceed to {display}, I need baseline data from a {fallback_test}"
        return {
            "type": "request_test",
            "test_name": fallback_test,
            "urgency": "normal",
            "utterance": self._format_test_request(fallback_test, reason),
            "rationale": reason,
        }

    def _format_test_request(self, test_name: str, reason: str) -> str:
        clean_name = self._resolve_test_display(test_name)
        request_token = clean_name.replace(" ", "_")
        short_reason = reason.strip().rstrip(".")
        if len(short_reason) > 120:
            short_reason = short_reason[:117].rstrip() + "..."
        return f"{short_reason}. REQUEST TEST: {request_token}"

    def _register_test(self, test_name: str) -> None:
        category, display = self._resolve_test_category(test_name)
        normalized = self._normalize_test_name(display)
        self.tests_requested.add(normalized)
        if category:
            self.completed_categories.add(category)

    def _resolve_test_category(self, test_name: Optional[str]) -> Tuple[Optional[str], str]:
        if not test_name:
            return None, ""
        normalized = self._normalize_test_name(test_name)
        if normalized in self.test_catalog:
            info = self.test_catalog[normalized]
            return info.get("category"), info.get("display", test_name)
        for key, info in self.test_catalog.items():
            if normalized in key or key in normalized:
                return info.get("category"), info.get("display", test_name)
        return None, test_name.strip()

    def _resolve_test_display(self, test_name: Optional[str]) -> str:
        return self._resolve_test_category(test_name)[1] if test_name else ""

    def _normalize_test_name(self, name: str) -> str:
        return re.sub(r"\s+", " ", name.replace("_", " ").replace("-", " ")).strip().lower()

    def _extract_test_from_utterance(self, utterance: str) -> Optional[str]:
        match = re.search(r"REQUEST TEST:\s*([A-Za-z0-9 _-]+)", utterance)
        if match:
            test_token = match.group(1).strip()
            return test_token.replace("_", " ").strip()
        return None

    def _call_section(self, section: str, payload: Dict[str, Any]) -> str:
        block = self.prompts.get(section)
        if not block:
            raise KeyError(f"Prompt section '{section}' is not defined")
        system_prompt = block.get("system", "")
        if self.bias_prompt:
            system_prompt = system_prompt + "\n\nBias context:\n" + self.bias_prompt
        user_template = block.get("user_template")
        if not user_template:
            raise KeyError(f"Prompt section '{section}' missing 'user_template'")
        payload_json = json.dumps(payload, ensure_ascii=False)
        user_prompt = user_template.format(payload=payload_json)
        return query_model(self.backend, user_prompt, system_prompt, scene=self.scenario)

    def _json_loads(self, raw: str) -> Any:
        raw = (raw or "").strip()
        if not raw:
            raise ValueError("Empty response from LLM")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for idx in range(len(raw)):
                try:
                    obj, _ = decoder.raw_decode(raw[idx:])
                    return obj
                except json.JSONDecodeError:
                    continue
        truncated = raw[:200] + ("..." if len(raw) > 200 else "")
        raise ValueError(f"Unable to parse JSON from response: {truncated}")

    def generate_bias(self) -> str:
        return self.bias_prompt

