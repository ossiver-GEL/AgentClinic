import json
from typing import Any, Dict, List, Optional, Set, Tuple

from llm import query_model
from agents.agent_tools import load_prompts_json


class EnhancedDoctorAgent:
    _TEST_CANONICAL: Dict[str, str] = {
        "chest_x-ray": "Chest_X-Ray",
        "chest_xray": "Chest_X-Ray",
        "chest_ct": "Chest_CT",
        "ct_chest": "Chest_CT",
        "ct_scan_thorax_and_abdomen": "CT_Scan_Thorax_and_Abdomen",
        "ct_scan_abdomen": "CT_Scan_Abdomen",
        "ct_abdomen": "CT_Scan_Abdomen",
        "ct_brain": "CT_Brain",
        "brain_ct": "CT_Brain",
        "head_ct": "CT_Brain",
        "ct_head_noncontrast": "CT_Head_Noncontrast",
        "head_ct_noncontrast": "CT_Head_Noncontrast",
        "mri_brain": "MRI_Brain",
        "brain_mri": "MRI_Brain",
        "mri_spine": "MRI_Spine",
        "spine_mri": "MRI_Spine",
        "barium_enema": "Barium_Enema",
        "abdominal_x-ray": "Abdominal_X-ray",
        "abdominal_xray": "Abdominal_X-ray",
        "abdominal_ultrasound": "Abdominal_Ultrasound",
        "ultrasound_abdomen": "Abdominal_Ultrasound",
        "ultrasound": "Ultrasound",
        "biopsy": "Biopsy",
        "lumbar_puncture": "Lumbar_Puncture",
        "ct_pulmonary_angiography": "CT_Pulmonary_Angiography",
        "pet_ct": "PET_CT",
        "echocardiogram": "Echocardiogram",
        "cardiac_catheterization": "Cardiac_Catheterization",
        "d-dimer": "D-Dimer",
        "ddimer": "D-Dimer",
        "fecal_occult_blood_test": "Fecal_Occult_Blood_Test",
        "urinalysis": "Urinalysis",
        "complete_blood_count": "Complete_Blood_Count",
        "basic_metabolic_panel": "Basic_Metabolic_Panel",
        "blood_culture": "Blood_Culture",
        "electrocardiogram": "Electrocardiogram",
        "ecg": "Electrocardiogram",
    }

    _TEST_PREDECESSOR_RULES: Dict[str, List[str]] = {
        "chest_ct": ["Chest_X-Ray"],
        "ct_chest": ["Chest_X-Ray"],
        "ct_scan_thorax_and_abdomen": ["Chest_X-Ray", "Abdominal_Ultrasound"],
        "ct_scan_abdomen": ["Abdominal_Ultrasound"],
        "ct_abdomen": ["Abdominal_Ultrasound"],
        "mri_brain": ["CT_Brain"],
        "brain_mri": ["CT_Brain"],
        "mri_spine": ["MRI_Brain"],
        "spine_mri": ["MRI_Brain"],
        "barium_enema": ["Abdominal_X-ray"],
        "biopsy": ["Ultrasound"],
        "lumbar_puncture": ["CT_Brain"],
        "ct_pulmonary_angiography": ["Chest_X-Ray", "D-Dimer"],
        "pet_ct": ["Chest_CT"],
        "cardiac_catheterization": ["Echocardiogram"],
        "angiography": ["Ultrasound"],
    }

    _LOW_COST_FALLBACK_TESTS: List[str] = [
        "Complete_Blood_Count",
        "Basic_Metabolic_Panel",
        "Urinalysis",
        "Chest_X-Ray",
        "Electrocardiogram",
    ]

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
        self.bias_present = (None if bias_present == "None" else bias_present)
        self.scenario = scenario
        self.img_request = img_request
        self.pipe = None

        self.prompts = load_prompts_json("enhanced_doctor")
        self.internal_prompts = self.prompts.get("internal", {})
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

        self.retry_limit = 3
        self.diagnosis_threshold = 0.95
        self.reconsider_threshold = 0.20
        self.max_reassessments = 2

        self.state_initialized = False
        self.differential: Dict[str, Any] = {}
        self.disease_confidence: Dict[str, float] = {}
        self.priority_queue: List[Dict[str, Any]] = []
        self.completed_tests: List[str] = []
        self.completed_tests_index: Set[str] = set()
        self.pending_test: Optional[Dict[str, Any]] = None
        self.questions_asked: List[str] = []
        self.observation_log: List[Dict[str, Any]] = []
        self.reassessments_done = 0
        self.last_action: Optional[Dict[str, Any]] = None

        self.reset()

    @staticmethod
    def _normalize_label(label: str) -> str:
        return label.replace("-", "_").replace(" ", "_").lower()

    def _canonical_test_name(self, name: str) -> str:
        key = self._normalize_label(name)
        return self._TEST_CANONICAL.get(key, name)

    def generate_bias(self) -> str:
        if self.bias_present is None:
            return ""
        prompts = self.prompts.get("biases", {})
        if self.bias_present in prompts:
            return prompts[self.bias_present]
        raise ValueError(f"Unsupported bias type: {self.bias_present}")

    def inference_doctor(self, patient_message: str, image_requested: bool = False) -> str:
        if self.infs >= self.MAX_INFS:
            raise RuntimeError("Maximum inferences reached")

        if not self.state_initialized:
            self._initialize_state()
            self.state_initialized = True

        cleaned_message = (patient_message or "").strip()
        if cleaned_message:
            self._process_patient_observation(cleaned_message)

        reply = self._choose_next_action(force_image_request=image_requested)

        self.agent_hist += patient_message + "\n\n" + reply + "\n\n"
        self.infs += 1
        self.last_action = {
            "text": reply,
            "turn": self.infs,
        }
        return reply

    def system_prompt(self) -> str:
        base = self.prompts["system_base"].format(self.MAX_INFS, self.infs)
        base_with_images = base + (self.prompts.get("system_images_suffix", "") if self.img_request else "")
        presentation = self.prompts["system_presentation_suffix"].format(self.presentation)
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        return base_with_images + bias_prompt + presentation

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()
        self.state_initialized = False
        self.differential = {}
        self.disease_confidence = {}
        self.priority_queue = []
        self.completed_tests = []
        self.completed_tests_index = set()
        self.pending_test = None
        self.questions_asked = []
        self.observation_log = []
        self.reassessments_done = 0
        self.last_action = None

    def _initialize_state(self) -> None:
        basic_info = self._serialize_context(self.presentation)
        known = self._invoke_internal(
            "feature_extraction",
            basic_info=basic_info,
        )
        diseases = self._invoke_internal(
            "differential",
            basic_info=basic_info,
            known_features=json.dumps(known, ensure_ascii=False),
        )
        table = self._invoke_internal(
            "feature_table",
            basic_info=basic_info,
            known_features=json.dumps(known, ensure_ascii=False),
            candidate_diseases=json.dumps(diseases, ensure_ascii=False),
        )
        self.differential = table
        self._recompute_confidence()
        self._refresh_priority_queue()

    def _serialize_context(self, payload: Any) -> str:
        if isinstance(payload, (dict, list)):
            return json.dumps(payload, ensure_ascii=False)
        return str(payload)

    def _invoke_internal(self, key: str, **kwargs) -> Dict[str, Any]:
        prompt_pack = self.internal_prompts.get(key)
        if not prompt_pack:
            raise KeyError(f"Missing internal prompt for key '{key}'")
        system_prompt = prompt_pack["system"]
        user_prompt = prompt_pack["user"].format(**kwargs)
        return self._call_model_json(system_prompt, user_prompt)

    def _call_model_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        attempts = 0
        prompt_variant = user_prompt
        last_error: Optional[Exception] = None
        while attempts < self.retry_limit:
            raw = query_model(self.backend, prompt_variant, system_prompt, scene=self.scenario)
            try:
                return self._parse_json(raw)
            except Exception as exc:
                last_error = exc
                attempts += 1
                if attempts >= self.retry_limit:
                    break
                prompt_variant = user_prompt + "\n\nRespond again using STRICT JSON with double quotes and no commentary."
        raise RuntimeError(f"Failed to obtain structured response: {last_error}")

    def _parse_json(self, raw: str) -> Dict[str, Any]:
        text = (raw or "").strip()
        if not text:
            raise ValueError("Empty response")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return {"data": parsed}
        except json.JSONDecodeError:
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start >= 0 and brace_end >= brace_start:
                snippet = text[brace_start: brace_end + 1]
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    return parsed
                return {"data": parsed}
            raise

    def _process_patient_observation(self, content: str) -> None:
        entry: Dict[str, Any] = {"raw": content}
        if content.upper().startswith("RESULTS:") or content.upper().startswith("NORMAL READINGS"):
            entry["type"] = "test_result"
            if self.pending_test:
                entry["test_name"] = self.pending_test.get("name")
                normalized = self._normalize_label(entry["test_name"])
                if normalized not in self.completed_tests_index:
                    self.completed_tests_index.add(normalized)
                    self.completed_tests.append(entry["test_name"])
            else:
                entry["test_name"] = None
        else:
            entry["type"] = "patient_answer"
        entry["turn"] = self.infs
        entry["question_context"] = self.last_action["text"] if self.last_action else ""

        update_payload = self._invoke_internal(
            "feature_update",
            observation=json.dumps(entry, ensure_ascii=False),
            differential=json.dumps(self.differential, ensure_ascii=False),
        )
        entry["summary"] = update_payload.get("summary", "")
        self._apply_feature_updates(update_payload)
        self.observation_log.append(entry)

        if update_payload.get("needs_reassessment") and self.reassessments_done < self.max_reassessments:
            self.reassessments_done += 1
            self._reassess_differential()

        self._recompute_confidence()
        self.pending_test = None

    def _apply_feature_updates(self, payload: Dict[str, Any]) -> None:
        updates = payload.get("updates") or []
        for upd in updates:
            disease_name = upd.get("disease")
            feature_name = upd.get("feature")
            if not disease_name or not feature_name:
                continue
            disease = self._find_disease(disease_name)
            if not disease:
                continue
            feature = self._find_feature(disease, feature_name)
            if not feature:
                feature = {
                    "name": feature_name,
                    "description": upd.get("description") or upd.get("evidence", ""),
                    "weight": max(float(upd.get("weight") or 0.3), 0.05),
                    "state": 0,
                    "collection": upd.get("collection") or "other",
                    "cost_level": upd.get("cost_level") or "medium",
                }
                disease.setdefault("features", []).append(feature)
            new_state = self._clamp_state(upd.get("new_state"))
            if new_state is not None:
                feature["state"] = new_state
            if upd.get("description"):
                feature["description"] = upd["description"]
        contradictions = payload.get("contradictions") or []
        if contradictions:
            for note in contradictions:
                disease = self._find_disease(note.get("disease"))
                feature = self._find_feature(disease, note.get("feature")) if disease else None
                if feature:
                    feature["state"] = self._clamp_state(note.get("new_state", -100))

    def _find_disease(self, name: str) -> Optional[Dict[str, Any]]:
        diseases = self.differential.get("diseases") or []
        name_lower = name.lower()
        for disease in diseases:
            if disease.get("name", "").lower() == name_lower:
                return disease
        return None

    def _find_feature(self, disease: Optional[Dict[str, Any]], feature_name: str) -> Optional[Dict[str, Any]]:
        if not disease:
            return None
        features = disease.get("features") or []
        target = feature_name.lower()
        for feat in features:
            if feat.get("name", "").lower() == target:
                return feat
        return None

    def _clamp_state(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError):
            return None
        return max(-100, min(100, numeric))

    def _recompute_confidence(self) -> None:
        confidences: Dict[str, float] = {}
        diseases = self.differential.get("diseases") or []
        for disease in diseases:
            features = disease.get("features") or []
            weights = [max(float(f.get("weight", 0.0)), 0.0) for f in features]
            total_weight = sum(weights)
            if total_weight <= 0.0:
                disease_conf = 0.0
            else:
                weighted = 0.0
                for idx, feat in enumerate(features):
                    state = self._clamp_state(feat.get("state")) or 0
                    weight_val = weights[idx]
                    weighted += weight_val * (state / 100.0)
                disease_conf = (weighted + total_weight) / (2.0 * total_weight)
                disease_conf = max(0.0, min(1.0, disease_conf))
            disease["confidence"] = round(disease_conf, 4)
            confidences[disease.get("name", "")] = disease_conf
        self.disease_confidence = confidences

    def _refresh_priority_queue(self) -> None:
        data = self._invoke_internal(
            "plan_generation",
            differential=json.dumps(self.differential, ensure_ascii=False),
            confidence=json.dumps(self.disease_confidence, ensure_ascii=False),
            observations=json.dumps(self._recent_observations(), ensure_ascii=False),
            completed_tests=json.dumps(self.completed_tests, ensure_ascii=False),
            remaining_turns=str(self.MAX_INFS - self.infs),
        )
        items = data.get("prioritized_items") or []
        self.priority_queue = [item for item in items if item]

    def _recent_observations(self, limit: int = 6) -> List[Dict[str, Any]]:
        recent = self.observation_log[-limit:]
        output = []
        for entry in recent:
            output.append(
                {
                    "type": entry.get("type"),
                    "summary": entry.get("summary") or entry.get("raw"),
                    "test_name": entry.get("test_name"),
                }
            )
        return output

    def _choose_next_action(self, force_image_request: bool = False) -> str:
        if self.infs >= self.MAX_INFS - 1:
            return self._final_diagnosis()

        best_name, best_conf = self._best_candidate()
        if best_name and best_conf >= self.diagnosis_threshold:
            return f"DIAGNOSIS READY: {best_name}"

        if (best_conf or 0.0) <= self.reconsider_threshold and self.reassessments_done < self.max_reassessments:
            self.reassessments_done += 1
            self._reassess_differential()
            best_name, best_conf = self._best_candidate()

        while self.priority_queue:
            item = self.priority_queue.pop(0)
            action_text = self._render_action(item, force_image_request)
            if action_text:
                return action_text

        self._refresh_priority_queue()
        if not self.priority_queue:
            return "Could you describe any other symptoms that concern you most right now?"
        return self._render_action(self.priority_queue.pop(0), force_image_request) or "Could you describe any other symptoms that concern you most right now?"

    def _render_action(self, item: Dict[str, Any], force_image_request: bool) -> Optional[str]:
        action_type = (item.get("action_type") or "").lower()
        if action_type == "test":
            return self._handle_test_action(item, force_image_request)
        if action_type == "ask":
            question = item.get("question_text") or item.get("dialogue") or ""
            question = question.strip()
            if not question:
                return None
            if question in self.questions_asked:
                return None
            self.questions_asked.append(question)
            return question
        if action_type == "diagnosis":
            candidate = item.get("diagnosis") or item.get("target_disease")
            if candidate:
                return f"DIAGNOSIS READY: {candidate}"
        return None

    def _handle_test_action(self, item: Dict[str, Any], force_image_request: bool) -> Optional[str]:
        requested = item.get("test_name") or ""
        requested = requested.strip()
        if not requested:
            return None
        canonical = self._canonical_test_name(requested)
        normalized = self._normalize_label(canonical)
        if normalized in self.completed_tests_index:
            return None

        urgent = bool(item.get("allow_skip_predecessors")) or force_image_request
        predecessor = self._select_predecessor(normalized, urgent)
        final_test = predecessor or canonical
        final_normalized = self._normalize_label(final_test)
        if final_normalized in self.completed_tests_index:
            return None

        self.pending_test = {
            "name": final_test,
            "original": canonical,
            "reason": item.get("reason"),
        }
        self.completed_tests_index.add(final_normalized)
        self.completed_tests.append(final_test)
        return f"REQUEST TEST: {final_test}"

    def _select_predecessor(self, normalized_test: str, urgent: bool) -> Optional[str]:
        reverse_lookup = {self._normalize_label(v): v for v in self._TEST_CANONICAL.values()}
        predecessors = self._TEST_PREDECESSOR_RULES.get(normalized_test) or []
        for pred in predecessors:
            pred_norm = self._normalize_label(pred)
            canonical = reverse_lookup.get(pred_norm, pred)
            if pred_norm not in self.completed_tests_index and not urgent:
                return canonical
        if not urgent:
            for fallback in self._LOW_COST_FALLBACK_TESTS:
                fallback_norm = self._normalize_label(fallback)
                if fallback_norm not in self.completed_tests_index:
                    return fallback
        return None

    def _best_candidate(self) -> Tuple[Optional[str], float]:
        if not self.disease_confidence:
            return None, 0.0
        best_name = max(self.disease_confidence, key=self.disease_confidence.get)
        return best_name, self.disease_confidence.get(best_name, 0.0)

    def _final_diagnosis(self) -> str:
        best_name, _ = self._best_candidate()
        if not best_name:
            return "DIAGNOSIS READY: Undifferentiated illness"
        return f"DIAGNOSIS READY: {best_name}"

    def _reassess_differential(self) -> None:
        basic_info = self._serialize_context(self.presentation)
        table = self._invoke_internal(
            "feature_table",
            basic_info=basic_info,
            known_features=json.dumps(self.differential, ensure_ascii=False),
            candidate_diseases=json.dumps({"diseases": list(self.disease_confidence.keys())}, ensure_ascii=False),
        )
        self.differential = table
        self._recompute_confidence()
        self._refresh_priority_queue()


__all__ = ["EnhancedDoctorAgent"]
