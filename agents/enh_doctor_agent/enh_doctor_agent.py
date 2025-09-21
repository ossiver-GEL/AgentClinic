import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from llm import query_model
from agents.agent_tools import load_prompts_json


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            child = _stringify(val)
            if child:
                parts.append(f"{key}: {child}")
        return "; ".join(parts)
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            child = _stringify(item)
            if child:
                parts.append(child)
        return "; ".join(parts)
    return str(value)


def _normalize_feature_key(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return cleaned.strip("_")


def _normalize_test_name(test_name: str) -> str:
    text = test_name.upper().replace("_", " ")
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _to_canonical_test_name(normalized: str) -> str:
    mapping = {
        "CHEST X-RAY": "Chest_X-Ray",
        "ABDOMINAL ULTRASOUND": "Abdominal_Ultrasound",
        "PELVIC ULTRASOUND": "Pelvic_Ultrasound",
        "CT HEAD": "CT_Head",
        "CT CHEST": "CT_Chest",
        "CT ABDOMEN": "CT_Abdomen",
        "CT ABDOMEN AND PELVIS": "CT_Abdomen_and_Pelvis",
        "MRI BRAIN": "MRI_Brain",
        "MRI HEAD": "MRI_Head",
        "MRI SPINE": "MRI_Spine",
        "MRI ABDOMEN": "MRI_Abdomen",
        "MRI PELVIS": "MRI_Pelvis",
        "ECHOCARDIOGRAM": "Echocardiogram",
        "ECG": "ECG",
        "CARDIAC CATHETERIZATION": "Cardiac_Catheterization",
        "COLONOSCOPY": "Colonoscopy",
        "FOBT": "FOBT",
        "UPPER ENDOSCOPY": "Upper_Endoscopy",
        "BARIUM SWALLOW": "Barium_Swallow",
        "BRONCHOSCOPY": "Bronchoscopy",
        "CT PULMONARY ANGIOGRAM": "CT_Pulmonary_Angiogram",
        "CTA CHEST": "CTA_Chest",
        "BONE SCAN": "Bone_Scan",
        "LUMBAR PUNCTURE": "Lumbar_Puncture",
        "ULTRASOUND": "Ultrasound",
        "ABDOMINAL CT": "CT_Abdomen",
        "MRI CHEST": "MRI_Chest",
    }
    if normalized in mapping:
        return mapping[normalized]
    return normalized.title().replace(" ", "_")


_TEST_CATEGORY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "basic_lab": (
        "CBC",
        "COMPLETE BLOOD COUNT",
        "BASIC METABOLIC PANEL",
        "BMP",
        "COMPREHENSIVE METABOLIC PANEL",
        "CMP",
        "ELECTROLYTE",
        "GLUCOSE",
        "URINALYSIS",
        "A1C",
        "HEMOGLOBIN",
        "TSH",
        "THYROID",
        "LIPID",
    ),
    "advanced_lab": (
        "AUTOANTIBODY",
        "TUMOR MARKER",
        "FLOW CYTOMETRY",
        "GENETIC",
        "PCR",
        "HLA",
    ),
    "basic_imaging": (
        "X-RAY",
        "XRAY",
        "ULTRASOUND",
        "US",
        "ECHO",
        "ECHOCARDIOGRAM",
        "ECG",
        "EKG",
    ),
    "advanced_imaging": (
        "CT",
        "MRI",
        "ANGIOGRAM",
        "CTA",
        "MRA",
        "PET",
        "SPECT",
    ),
    "functional_imaging": (
        "NUCLEAR",
        "PERFUSION",
        "VENTILATION",
        "MUGA",
    ),
    "invasive_procedure": (
        "BIOPSY",
        "ENDOSCOPY",
        "COLONOSCOPY",
        "LAPAROSCOPY",
        "LUMBAR PUNCTURE",
        "BRONCHOSCOPY",
        "CATHETERIZATION",
        "ANGIOGRAPHY",
        "ARTHROSCOPY",
    ),
}


_DEFAULT_TEST_RELATIONSHIPS: Dict[str, List[str]] = {
    "CT CHEST": ["CHEST X-RAY"],
    "CT PULMONARY ANGIOGRAM": ["CHEST X-RAY"],
    "CTA CHEST": ["CHEST X-RAY"],
    "CT ABDOMEN": ["ABDOMINAL ULTRASOUND"],
    "CT ABDOMEN AND PELVIS": ["ABDOMINAL ULTRASOUND"],
    "MRI ABDOMEN": ["ABDOMINAL ULTRASOUND"],
    "MRI PELVIS": ["PELVIC ULTRASOUND"],
    "MRI BRAIN": ["CT HEAD"],
    "MRI HEAD": ["CT HEAD"],
    "MRI SPINE": ["SPINE X-RAY"],
    "PET SCAN": ["CT CHEST"],
    "BONE SCAN": ["CHEST X-RAY"],
    "CARDIAC CATHETERIZATION": ["ECHOCARDIOGRAM", "ECG"],
    "LUMBAR PUNCTURE": ["CT HEAD"],
    "COLONOSCOPY": ["FOBT"],
    "UPPER ENDOSCOPY": ["BARIUM SWALLOW"],
    "BRONCHOSCOPY": ["CHEST X-RAY"],
}


_CATEGORY_PREREQS: Dict[str, Tuple[str, ...]] = {
    "advanced_lab": ("basic_lab",),
    "advanced_imaging": ("basic_imaging",),
    "functional_imaging": ("basic_imaging", "advanced_imaging"),
    "invasive_procedure": ("basic_imaging", "advanced_imaging"),
}


class EnhancedDoctorAgent:
    CONFIRM_THRESHOLD = 0.9
    RECONSIDER_THRESHOLD = 0.05
    MIN_FEATURES_FOR_DIAGNOSIS = 2
    MIN_COVERAGE_FOR_DIAGNOSIS = 0.55
    MAX_CONTEXT_EVENTS = 18
    MAX_LLM_RETRIES = 3
    MAX_RECONSIDER_ATTEMPTS = 2

    def __init__(self, scenario, backend_str="gpt-4o-mini", max_infs=20, bias_present=None, img_request=False) -> None:
        self.infs = 0
        self.MAX_INFS = max_infs
        self.agent_hist = ""
        self.backend = backend_str
        self.bias_present = (None if bias_present == "None" else bias_present)
        self.scenario = scenario
        self.img_request = img_request
        self.pipe = None

        self.prompts = load_prompts_json("enh_doctor")
        self.max_llm_retries = self.prompts.get("settings", {}).get("max_json_retries", self.MAX_LLM_RETRIES)

        self.reset()

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()
        self.presentation_str = _stringify(self.presentation)
        self.bias_text = self.generate_bias()

        self.knowledge_events: List[Dict[str, Any]] = []
        self.typical_features: List[Dict[str, Any]] = []
        self.hypotheses: List[Dict[str, Any]] = []
        self.feature_status: Dict[str, Dict[str, Any]] = {}
        self.plan: List[Dict[str, Any]] = []
        self.feature_attempts: Dict[str, int] = {}
        self.tests_completed: Dict[str, Dict[str, Any]] = {}
        self.pending_test: Optional[Dict[str, Any]] = None
        self.needs_replan = True
        self.reconsider_attempts = 0
        self.last_action: Optional[Dict[str, Any]] = None
        self.last_patient_message = ""

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

        cleaned_question = (question or "").strip()
        if cleaned_question:
            self._record_patient_message(cleaned_question)
            self.needs_replan = True
        elif not self.hypotheses:
            self.needs_replan = True

        final_turn = (self.MAX_INFS - self.infs) == 1

        if self.needs_replan:
            self._recompute_differential(force_broaden=False)
            self.needs_replan = False

        if final_turn and not self.hypotheses:
            self._recompute_differential(force_broaden=True)

        if self.hypotheses and self.hypotheses[0]["confidence"] <= self.RECONSIDER_THRESHOLD and self.reconsider_attempts < self.MAX_RECONSIDER_ATTEMPTS:
            self.reconsider_attempts += 1
            self._recompute_differential(force_broaden=True)

        action = self._choose_next_action(final_turn=final_turn)
        response = self._render_action(action)

        self.agent_hist += (cleaned_question + "\n\n" if cleaned_question else "") + response + "\n\n"
        self.last_action = action
        self.infs += 1
        return response

    def _record_patient_message(self, message: str) -> None:
        entry_type = "patient_response"
        if self.pending_test:
            entry_type = "test_result"
        event = {
            "type": entry_type,
            "content": message,
            "turn": self.infs,
        }
        if self.pending_test:
            test_name = self.pending_test.get("name")
            event["test"] = test_name
            normalized = _normalize_test_name(test_name)
            self.tests_completed[normalized] = {
                "name": test_name,
                "result": message,
            }
            self.pending_test = None
        self.knowledge_events.append(event)
        if len(self.knowledge_events) > self.MAX_CONTEXT_EVENTS:
            self.knowledge_events = self.knowledge_events[-self.MAX_CONTEXT_EVENTS:]
        self.last_patient_message = message

    # ----------------
    # Planning helpers
    # ----------------

    def _recompute_differential(self, force_broaden: bool) -> None:
        context_summary = self._build_context_summary()
        features_response = self._call_llm_json(
            "feature_extraction",
            {
                "presentation": self.presentation_str,
                "known_facts": context_summary,
                "recent_message": self.last_patient_message or "None",
            },
        )
        features = features_response.get("typical_features")
        if not isinstance(features, list):
            raise ValueError("feature_extraction did not return 'typical_features' list")
        self.typical_features = features

        disease_response = self._call_llm_json(
            "disease_hypotheses",
            {
                "presentation": self.presentation_str,
                "known_facts": context_summary,
                "feature_candidates": json.dumps(self.typical_features, ensure_ascii=False),
                "prior_diseases": json.dumps([h.get("name") for h in self.hypotheses], ensure_ascii=False),
                "force_broaden": str(force_broaden).lower(),
            },
        )
        disease_candidates = disease_response.get("possible_diseases")
        if not isinstance(disease_candidates, list) or not disease_candidates:
            raise ValueError("disease_hypotheses did not return candidates")

        matrix_response = self._call_llm_json(
            "disease_feature_matrix",
            {
                "presentation": self.presentation_str,
                "known_facts": context_summary,
                "feature_candidates": json.dumps(self.typical_features, ensure_ascii=False),
                "candidate_diseases": json.dumps(disease_candidates, ensure_ascii=False),
            },
        )
        disease_entries = matrix_response.get("diseases")
        if not isinstance(disease_entries, list) or not disease_entries:
            raise ValueError("disease_feature_matrix did not return 'diseases'")

        self._ingest_feature_matrix(disease_entries)

        plan_response = self._call_llm_json(
            "feature_prioritization",
            {
                "known_facts": context_summary,
                "feature_matrix": json.dumps(disease_entries, ensure_ascii=False),
                "disease_confidence": json.dumps(
                    [
                        {
                            "name": hyp["name"],
                            "confidence": hyp["confidence"],
                            "coverage": hyp["covered_weight"] / hyp["total_weight"] if hyp["total_weight"] else 0.0,
                        }
                        for hyp in self.hypotheses
                    ],
                    ensure_ascii=False,
                ),
                "asked_features": json.dumps(
                    [
                        {"feature": name, "attempts": count}
                        for name, count in self.feature_attempts.items()
                    ],
                    ensure_ascii=False,
                ),
            },
        )
        prioritized = plan_response.get("prioritized_features", [])
        if not isinstance(prioritized, list):
            raise ValueError("feature_prioritization did not return list")
        self.plan = prioritized

    def _build_context_summary(self) -> str:
        lines = []
        if self.bias_text:
            lines.append(f"Bias context: {self.bias_text.strip()}")
        for idx, event in enumerate(self.knowledge_events[-self.MAX_CONTEXT_EVENTS:]):
            prefix = "Patient" if event["type"] == "patient_response" else "Test"
            if event["type"] == "test_result":
                test_name = event.get("test", "Unknown test")
                lines.append(f"Test {test_name}: {event['content']}")
            else:
                lines.append(f"Patient response: {event['content']}")
            if idx >= self.MAX_CONTEXT_EVENTS:
                break
        if not lines:
            lines.append("No responses collected yet.")
        return "\n".join(lines)

    def _ingest_feature_matrix(self, disease_entries: List[Dict[str, Any]]) -> None:
        feature_status: Dict[str, Dict[str, Any]] = {}
        hypotheses: List[Dict[str, Any]] = []

        for disease in disease_entries:
            name = disease.get("name")
            if not name:
                continue
            features = disease.get("features", [])
            if not isinstance(features, list):
                raise ValueError("Each disease must include feature list")
            total_weight = 0.0
            covered_weight = 0.0
            weighted_score = 0.0
            supporting = []
            contradicting = []

            parsed_features = []
            for feature in features:
                feat_name = feature.get("feature")
                if not feat_name:
                    continue
                weight = float(feature.get("weight", 0.0))
                weight = max(0.0, min(1.0, weight))
                match = int(float(feature.get("match", 0)))
                match = max(-100, min(100, match))
                collection = feature.get("collection_method", "question")
                note = feature.get("notes", "")
                expected_positive = feature.get("expected_positive", "")
                total_weight += weight
                if abs(match) > 0:
                    covered_weight += weight
                weighted_score += weight * (match / 100.0)
                if match >= 50:
                    supporting.append(feat_name)
                elif match <= -50:
                    contradicting.append(feat_name)

                key = _normalize_feature_key(feat_name)
                status_entry = feature_status.setdefault(
                    key,
                    {
                        "name": feat_name,
                        "collection_method": collection,
                        "related_diseases": [],
                        "matches": [],
                        "weights": [],
                        "notes": [],
                    },
                )
                status_entry["collection_method"] = collection or status_entry.get("collection_method", "question")
                status_entry["related_diseases"].append(name)
                status_entry["matches"].append({"disease": name, "match": match, "weight": weight})
                status_entry["weights"].append(weight)
                if note:
                    status_entry["notes"].append(note)
                if expected_positive:
                    status_entry.setdefault("expected_positive", set()).add(expected_positive)

                parsed_features.append(
                    {
                        "feature": feat_name,
                        "weight": weight,
                        "match": match,
                        "collection_method": collection,
                        "notes": note,
                        "expected_positive": expected_positive,
                    }
                )

            confidence = self._calculate_confidence(weighted_score, total_weight, covered_weight)
            hypotheses.append(
                {
                    "name": name,
                    "features": parsed_features,
                    "confidence": confidence,
                    "weighted_score": weighted_score,
                    "total_weight": total_weight,
                    "covered_weight": covered_weight,
                    "supporting": supporting,
                    "contradicting": contradicting,
                }
            )

        if not hypotheses:
            raise ValueError("No hypotheses produced from feature matrix")

        self.feature_status = feature_status
        self.hypotheses = sorted(hypotheses, key=lambda item: item["confidence"], reverse=True)

    def _calculate_confidence(self, weighted_score: float, total_weight: float, covered_weight: float) -> float:
        if total_weight <= 0:
            return 0.0
        normalized = 0.5 * ((weighted_score / total_weight) + 1.0)
        normalized = max(0.0, min(1.0, normalized))
        coverage_ratio = covered_weight / total_weight if total_weight else 0.0
        # dampen confidence if little evidence collected
        adjusted = normalized * (0.2 + 0.8 * math.sqrt(coverage_ratio))
        return max(0.0, min(1.0, adjusted))

    # -----------------
    # Action selection
    # -----------------

    def _choose_next_action(self, final_turn: bool) -> Dict[str, Any]:
        if not self.hypotheses:
            return {"type": "diagnosis", "force": True, "disease": "Undifferentiated condition"}

        top = self.hypotheses[0]
        coverage = top["covered_weight"] / top["total_weight"] if top["total_weight"] else 0.0
        if final_turn:
            return self._make_diagnosis_action(force=True)

        if (
            top["confidence"] >= self.CONFIRM_THRESHOLD
            and coverage >= self.MIN_COVERAGE_FOR_DIAGNOSIS
            and len(top["supporting"]) >= self.MIN_FEATURES_FOR_DIAGNOSIS
        ):
            return self._make_diagnosis_action(force=False)

        for item in self.plan:
            feature_name = item.get("feature")
            action_type = (item.get("action_type") or "").lower()
            if not feature_name or action_type not in {"question", "test", "exam"}:
                continue
            status = self.feature_status.get(_normalize_feature_key(feature_name))
            if status and self._feature_resolved(status):
                continue
            return self._action_from_plan_item(item, status)

        if self.reconsider_attempts < self.MAX_RECONSIDER_ATTEMPTS:
            self.reconsider_attempts += 1
            self._recompute_differential(force_broaden=True)
            return self._choose_next_action(final_turn=final_turn)

        return self._make_diagnosis_action(force=False)

    def _feature_resolved(self, status: Optional[Dict[str, Any]]) -> bool:
        if not status:
            return False
        matches = status.get("matches", [])
        if not matches:
            return False
        return all(entry.get("match") not in (0, None) for entry in matches)

    def _action_from_plan_item(self, item: Dict[str, Any], status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        action_type = (item.get("action_type") or "question").lower()
        feature_name = item.get("feature", "")
        feature_key = _normalize_feature_key(feature_name)
        if action_type == "test":
            recommended_test = item.get("recommended_test") or feature_name
            hierarchy = self._apply_test_hierarchy(recommended_test, item)
            selected_test = hierarchy["test"]
            reason = hierarchy.get("reason", item.get("reason", ""))
            if hierarchy.get("is_prereq"):
                note = f"Prerequisite before {recommended_test}: {selected_test}."
            else:
                note = reason
            return {
                "type": "test",
                "test_name": selected_test,
                "feature": feature_name,
                "note": note,
                "original_test": recommended_test,
                "feature_key": feature_key,
            }
        if action_type == "exam":
            # treat as question-style physical exam prompt
            question_text = self._generate_question_for_feature(feature_name, item, status, request_physical=True)
            return {
                "type": "question",
                "question": question_text,
                "feature": feature_name,
                "feature_key": feature_key,
            }
        question_text = self._generate_question_for_feature(feature_name, item, status)
        return {
            "type": "question",
            "question": question_text,
            "feature": feature_name,
            "feature_key": feature_key,
        }

    def _make_diagnosis_action(self, force: bool) -> Dict[str, Any]:
        if not self.hypotheses:
            return {"type": "diagnosis", "force": True, "disease": "Undetermined condition"}
        top = self.hypotheses[0]
        disease = top.get("name", "Undetermined condition")
        return {
            "type": "diagnosis",
            "disease": disease,
            "force": force,
        }

    def _generate_question_for_feature(
        self,
        feature_name: str,
        plan_item: Dict[str, Any],
        status: Optional[Dict[str, Any]],
        request_physical: bool = False,
    ) -> str:
        related = ", ".join(status.get("related_diseases", [])) if status else ""
        payload = {
            "feature_name": feature_name,
            "reason": plan_item.get("reason", ""),
            "related_diseases": related or plan_item.get("related_diseases", ""),
            "known_facts": self._build_recent_fact_string(),
            "bias_note": self.bias_text or "None",
            "collection_method": "exam" if request_physical else plan_item.get("collection_method", "question"),
        }
        response = self._call_llm_json("question_generation", payload)
        question_text = response.get("question")
        if not question_text:
            raise ValueError("question_generation did not supply question")
        question_text = question_text.strip()
        if not question_text.endswith("?"):
            question_text = question_text.rstrip(".") + "?"
        feature_key = _normalize_feature_key(feature_name)
        self.feature_attempts[feature_key] = self.feature_attempts.get(feature_key, 0) + 1
        return question_text

    def _build_recent_fact_string(self, limit: int = 5) -> str:
        events = self.knowledge_events[-limit:]
        if not events:
            return "No additional findings yet."
        lines = []
        for event in events:
            if event["type"] == "test_result":
                test_name = event.get("test", "Test")
                lines.append(f"{test_name}: {event['content']}")
            else:
                lines.append(event["content"])
        return " | ".join(lines)

    # -------------------
    # Test hierarchy logic
    # -------------------

    def _apply_test_hierarchy(self, requested_test: str, plan_item: Dict[str, Any]) -> Dict[str, Any]:
        normalized = _normalize_test_name(requested_test)
        category = self._categorize_test(normalized)
        suggested_prereqs = [
            _normalize_test_name(item)
            for item in plan_item.get("suggested_prerequisites", [])
            if isinstance(item, str)
        ]
        default_prereqs = _DEFAULT_TEST_RELATIONSHIPS.get(normalized, [])
        category_prereqs = []
        for prereq_cat in _CATEGORY_PREREQS.get(category, ()): 
            fallback = self._fallback_test_for_category(normalized, prereq_cat)
            if fallback:
                category_prereqs.append(_normalize_test_name(fallback))

        all_prereqs = []
        seen = set()
        for seq in (suggested_prereqs, default_prereqs, category_prereqs):
            for item in seq:
                if item and item not in seen:
                    seen.add(item)
                    all_prereqs.append(item)

        urgency = (plan_item.get("urgency") or "routine").lower()
        for prereq in all_prereqs:
            if prereq not in self.tests_completed:
                if urgency in {"high", "critical"}:
                    continue
                canonical = _to_canonical_test_name(prereq)
                reason = f"Completing {canonical} first aligns with the staged diagnostic workflow."
                return {
                    "test": canonical,
                    "reason": reason,
                    "is_prereq": True,
                }

        canonical_requested = _to_canonical_test_name(normalized)
        return {
            "test": canonical_requested,
            "reason": plan_item.get("reason", ""),
            "is_prereq": False,
        }

    def _categorize_test(self, normalized_name: str) -> str:
        for category, keywords in _TEST_CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in normalized_name:
                    return category
        return "other"

    def _fallback_test_for_category(self, normalized_name: str, category: str) -> Optional[str]:
        if category == "basic_lab":
            return "CBC"
        if category == "basic_imaging":
            if "CHEST" in normalized_name:
                return "Chest_X-Ray"
            if any(word in normalized_name for word in ("ABDOM", "GI")):
                return "Abdominal_Ultrasound"
            if any(word in normalized_name for word in ("PELV", "OB", "GYNE")):
                return "Pelvic_Ultrasound"
            if any(word in normalized_name for word in ("HEAD", "BRAIN")):
                return "CT_Head"
            return "Ultrasound"
        if category == "advanced_imaging":
            if any(word in normalized_name for word in ("HEAD", "BRAIN")):
                return "CT_Head"
            if "CHEST" in normalized_name:
                return "Chest_X-Ray"
            if any(word in normalized_name for word in ("ABDOM", "GI")):
                return "Abdominal_Ultrasound"
            return "Chest_X-Ray"
        if category == "functional_imaging":
            return "CT_Chest"
        if category == "advanced_lab":
            return "CBC"
        if category == "invasive_procedure":
            if any(word in normalized_name for word in ("GI", "COLON", "GAST", "ESOPH")):
                return "Abdominal_Ultrasound"
            if any(word in normalized_name for word in ("CARD", "HEART")):
                return "Echocardiogram"
            if any(word in normalized_name for word in ("CHEST", "PULM", "LUNG")):
                return "Chest_X-Ray"
            if any(word in normalized_name for word in ("NEURO", "BRAIN", "SPINE")):
                return "CT_Head"
            return "Chest_X-Ray"
        return None

    # --------------
    # LLM utilities
    # --------------

    def _call_llm_json(self, prompt_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        workflow = self.prompts.get("workflow", {})
        if prompt_key not in workflow:
            raise ValueError(f"Prompt configuration for '{prompt_key}' is missing")
        config = workflow[prompt_key]
        system_prompt = config.get("system")
        user_template = config.get("user_template")
        if not system_prompt or not user_template:
            raise ValueError(f"Prompt '{prompt_key}' missing system or user template")
        try:
            user_prompt = user_template.format(**payload)
        except KeyError as exc:
            raise ValueError(f"Missing payload key {exc} for prompt '{prompt_key}'") from exc

        last_error: Optional[Exception] = None
        for attempt in range(int(self.max_llm_retries)):
            suffix = "" if attempt == 0 else "\nPlease respond with STRICT JSON only."
            raw = query_model(self.backend, user_prompt + suffix, system_prompt, scene=self.scenario)
            try:
                return self._parse_json_response(raw)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Failed to parse JSON from LLM for '{prompt_key}': {last_error}")

    @staticmethod
    def _parse_json_response(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    # --------------
    # Response render
    # --------------

    def _render_action(self, action: Dict[str, Any]) -> str:
        action_type = action.get("type")
        if action_type == "diagnosis":
            disease = action.get("disease") or "Undetermined condition"
            return f"DIAGNOSIS READY: {disease}"
        if action_type == "test":
            test_name = action["test_name"]
            self.pending_test = {"name": test_name, "feature": action.get("feature")}
            note = action.get("note", "")
            if note:
                note = note.strip()
                return f"{note} REQUEST TEST: {test_name}"
            return f"REQUEST TEST: {test_name}"
        if action_type == "question":
            return action["question"]
        raise ValueError(f"Unknown action type: {action_type}")

