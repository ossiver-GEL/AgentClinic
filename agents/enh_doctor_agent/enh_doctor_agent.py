from llm import query_model
from agents.agent_tools import load_prompts_json
import json
import re
from typing import List, Dict, Any, Optional


class EnhancedDoctorAgent:
    def __init__(self, scenario, backend_str: str = "gpt-4o-mini", max_infs: int = 20,
                 bias_present: Optional[str] = None, img_request: bool = False,
                 max_hypotheses: int = 5) -> None:
        # number of inference calls to the doctor
        self.infs = 0
        # maximum number of inference calls to the doctor
        self.MAX_INFS = max_infs
        # conversation history between doctor and patient (plain text log for output record)
        self.agent_hist = ""
        # presentation information for doctor
        self.presentation = ""
        # language model backend for doctor agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.pipe = None
        self.img_request = img_request
        self.biases = [
            "recency", "frequency", "false_consensus", "confirmation", "status_quo",
            "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"
        ]

        # Enhanced reasoning state
        self.knowledge_pairs: List[Dict[str, str]] = []  # list of {action, result}
        self.last_action: Optional[str] = None  # last outward action text (question or REQUEST TEST...)
        self.max_hypotheses = max_hypotheses

        # init
        self.reset()

    # =========================
    # Public API (called by runner)
    # =========================
    def inference_doctor(self, question: str, image_requested: bool = False) -> str:
        """Main outward turn. Implements the enhanced internal loop per design while keeping the external protocol.

        - Do NOT change information acquisition pattern. Only optimize internal processing.
        - Always return exactly one of:
          * a single short question (string),
          * 'REQUEST TEST: <Test_Name>',
          * 'REQUEST IMAGES' (if allowed),
          * 'DIAGNOSIS READY: <disease>'.
        - Errors must raise and not be silently ignored.
        """
        if self.infs >= self.MAX_INFS:
            return "Maximum inferences reached"

        # 0) Integrate previous action result into structured knowledge (condensation) if applicable
        self._integrate_last_result(question)

        # 1) Generate hypotheses set H from current knowledge
        I_text = self._format_I_text()
        hypotheses = self._gen_hypotheses(I_text, image_requested=image_requested)
        if not hypotheses:
            raise RuntimeError("No hypotheses returned by LLM; cannot proceed.")

        # Mandatory: if last remaining turn, must output a diagnosis immediately
        if self.infs == self.MAX_INFS - 1:
            answer = f"DIAGNOSIS READY: {hypotheses[0]}"
            self._finalize_and_log(question, answer)
            return answer

        # 2/3) Single vs multiple hypotheses analysis
        if len(hypotheses) == 1:
            d = hypotheses[0]
            analysis = self._analyze_single(I_text, d, image_requested=image_requested)
            pending: List[str] = _ensure_list(analysis.get("pending_evidence"))
            # 6) Termination: if unique disease and nothing pending, finalize
            if all((not isinstance(p, str)) or (len(p.strip()) == 0) for p in pending) or len(pending) == 0:
                answer = f"DIAGNOSIS READY: {d}"
                self._finalize_and_log(question, answer)
                return answer
            # 4) Propose next best action to get pending evidence
            need = f"Clarify remaining evidence for {d}: {', '.join([p for p in pending if isinstance(p, str)][:3])}"
            action = self._propose_next_action(I_text, need, allow_images=self.img_request, image_available=image_requested)
        else:
            d1, d2 = hypotheses[0], hypotheses[1]
            dual = self._analyze_dual(I_text, d1, d2, image_requested=image_requested)
            missing: List[str] = _ensure_list(dual.get("key_missing_info"))
            need = f"Differentiate between {d1} and {d2}; missing={', '.join([m for m in missing if isinstance(m, str)][:3])}"
            # 4) Propose next best action to differentiate top two
            action = self._propose_next_action(I_text, need, allow_images=self.img_request, image_available=image_requested)

        # 5) Execute outward: format as required and update last_action
        answer = self._format_outward_action(action)
        if not isinstance(answer, str) or len(answer.strip()) == 0:
            raise RuntimeError("Proposed action is empty; cannot proceed.")

        # Track last action for next turn integration
        if answer.startswith("REQUEST TEST:"):
            self.last_action = answer
        elif answer.strip() == "REQUEST IMAGES":
            self.last_action = "REQUEST IMAGES"
        elif answer.startswith("DIAGNOSIS READY"):
            self.last_action = "Final diagnosis"
        else:
            # treat as the exact question text
            self.last_action = answer

        # Log and step
        self._finalize_and_log(question, answer)
        return answer

    def reset(self) -> None:
        # Reset external log
        self.agent_hist = ""
        # Seed presentation from scenario
        self.presentation = self.scenario.examiner_information()
        # Reset enhanced state
        self.knowledge_pairs = []
        self.last_action = None
        # Add initial known info pair from presentation
        if isinstance(self.presentation, str) and self.presentation.strip():
            self.knowledge_pairs.append({
                "action": "Examiner information/context provided",
                "result": self.presentation.strip(),
            })

    # =========================
    # Internal helpers
    # =========================
    def _internal_prompts(self) -> Dict[str, Any]:
        prompts = load_prompts_json("enhanced_doctor")
        internal = prompts.get("internal")
        if not internal or not isinstance(internal, dict):
            raise FileNotFoundError("Missing 'internal' prompts in enh_doctor.json.")
        return internal

    def _system_bias_str(self) -> str:
        if self.bias_present is None:
            return ""
        prompts = load_prompts_json("enhanced_doctor").get("biases", {})
        return prompts.get(self.bias_present, "")

    def _integrate_last_result(self, result_text: str) -> None:
        """Condense last action and its result into a compact {o,i} pair and add to I."""
        if not self.last_action:
            return
        if not result_text or not isinstance(result_text, str) or len(result_text.strip()) == 0:
            return
        # Filter out run control marker if present
        cleaned = result_text.replace("This is the final question. Please provide a diagnosis.", "").strip()
        if len(cleaned) == 0:
            return
        internal = self._internal_prompts()
        sys = (internal.get("sys") or "You are a clinical reasoning assistant. Output only strict JSON.")
        sys = self._append_bias_to_system(sys)
        tmpl = internal.get("condense_oi")
        if not tmpl:
            raise FileNotFoundError("Missing 'condense_oi' in internal prompts.")
        prompt = tmpl.format(o=self.last_action, i=cleaned)
        resp = query_model(self.backend, prompt, sys)
        data = _parse_json_or_raise(resp)
        action = data.get("action")
        result = data.get("result")
        if not (isinstance(action, str) and isinstance(result, str)):
            raise ValueError("Invalid condense_oi JSON: expected 'action' and 'result' strings.")
        self.knowledge_pairs.append({"action": action.strip(), "result": result.strip()})

    def _format_I_text(self) -> str:
        parts = []
        for idx, pair in enumerate(self.knowledge_pairs, start=1):
            a = pair.get("action", "").strip()
            r = pair.get("result", "").strip()
            if a or r:
                parts.append(f"{idx}. Action: {a}; Result: {r}")
        return "\n".join(parts) if parts else "(no info)"

    def _gen_hypotheses(self, I_text: str, image_requested: bool) -> List[str]:
        internal = self._internal_prompts()
        sys = (internal.get("sys") or "You are a clinical reasoning assistant. Output only strict JSON.")
        sys = self._append_bias_to_system(sys)
        tmpl = internal.get("hypothesis")
        if not tmpl:
            raise FileNotFoundError("Missing 'hypothesis' in internal prompts.")
        prompt = tmpl.format(I=I_text, max_h=self.max_hypotheses)
        resp = query_model(self.backend, prompt, sys, image_requested=image_requested, scene=self.scenario)
        data = _parse_json_or_raise(resp)
        hyps = data.get("hypotheses")
        if not isinstance(hyps, list) or not hyps:
            raise ValueError("Invalid hypothesis JSON: 'hypotheses' must be a non-empty list.")
        # keep unique while preserving order; strip empties
        seen = set()
        out: List[str] = []
        for h in hyps:
            if isinstance(h, str):
                hs = h.strip()
                if hs and hs.lower() not in seen:
                    seen.add(hs.lower())
                    out.append(hs)
        if not out:
            raise ValueError("No valid hypotheses parsed after cleaning.")
        return out[: self.max_hypotheses]

    def _analyze_single(self, I_text: str, d: str, image_requested: bool) -> Dict[str, Any]:
        internal = self._internal_prompts()
        sys = (internal.get("sys") or "You are a clinical reasoning assistant. Output only strict JSON.")
        sys = self._append_bias_to_system(sys)
        tmpl = internal.get("analyze_single")
        if not tmpl:
            raise FileNotFoundError("Missing 'analyze_single' in internal prompts.")
        prompt = tmpl.format(I=I_text, d=d)
        resp = query_model(self.backend, prompt, sys, image_requested=image_requested, scene=self.scenario)
        return _parse_json_or_raise(resp)

    def _analyze_dual(self, I_text: str, d1: str, d2: str, image_requested: bool) -> Dict[str, Any]:
        internal = self._internal_prompts()
        sys = (internal.get("sys") or "You are a clinical reasoning assistant. Output only strict JSON.")
        sys = self._append_bias_to_system(sys)
        tmpl = internal.get("analyze_dual")
        if not tmpl:
            raise FileNotFoundError("Missing 'analyze_dual' in internal prompts.")
        prompt = tmpl.format(I=I_text, d1=d1, d2=d2)
        resp = query_model(self.backend, prompt, sys, image_requested=image_requested, scene=self.scenario)
        return _parse_json_or_raise(resp)

    def _propose_next_action(self, I_text: str, need: str, allow_images: bool, image_available: bool) -> Dict[str, Any]:
        internal = self._internal_prompts()
        sys = (internal.get("sys") or "You are a clinical reasoning assistant. Output only strict JSON.")
        sys = self._append_bias_to_system(sys)
        tmpl = internal.get("next_action")
        if not tmpl:
            raise FileNotFoundError("Missing 'next_action' in internal prompts.")
        # Provide explicit context about image availability to steer LLM
        images_hint = ("Images allowed; currently available" if (allow_images and image_available)
                       else ("Images allowed; not yet available" if allow_images else "Images not allowed"))
        need_text = f"{need} ({images_hint})"
        prompt = tmpl.format(I=I_text, need=need_text)
        resp = query_model(self.backend, prompt, sys, image_requested=image_available, scene=self.scenario)
        data = _parse_json_or_raise(resp)
        at = data.get("action_type")
        if at not in ("ask", "test", "diagnose", "images"):
            raise ValueError(f"Invalid action_type: {at}")
        if at == "ask":
            if not isinstance(data.get("content"), str) or not data["content"].strip():
                raise ValueError("Action 'ask' requires non-empty 'content'.")
        if at == "test":
            if not isinstance(data.get("content"), str) or not data["content"].strip():
                raise ValueError("Action 'test' requires non-empty 'content'.")
        if at == "diagnose":
            if not isinstance(data.get("diagnosis"), str) or not data["diagnosis"].strip():
                raise ValueError("Action 'diagnose' requires non-empty 'diagnosis'.")
        if at == "images" and not allow_images:
            # If images not allowed, this is an invalid response
            raise ValueError("Images requested but images are not allowed in this scenario.")
        return data

    def _format_outward_action(self, action_json: Dict[str, Any]) -> str:
        at = action_json.get("action_type")
        if at == "ask":
            # Return just the question text
            return action_json.get("content", "").strip()
        if at == "test":
            return f"REQUEST TEST: {action_json.get('content', '').strip()}"
        if at == "images":
            return "REQUEST IMAGES"
        if at == "diagnose":
            return f"DIAGNOSIS READY: {action_json.get('diagnosis', '').strip()}"
        raise ValueError(f"Unsupported action_type: {at}")

    def _finalize_and_log(self, question: str, answer: str) -> None:
        # Keep the plain-text agent history for auditing/output
        self.agent_hist += (question or "") + "\n\n" + answer + "\n\n"
        self.infs += 1

    def _append_bias_to_system(self, sys_prompt: str) -> str:
        bias = self._system_bias_str().strip()
        if bias:
            return sys_prompt + "\n\nCognitive constraint (may affect reasoning):\n" + bias
        return sys_prompt


# =========================
# Utilities (local only)
# =========================
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")
_JSON_ARR_RE = re.compile(r"\[[\s\S]*\]")


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    # Extract content inside ```json ... ``` if present
    m = re.search(r"^```(?:json|JSON)?\s*\n([\s\S]*?)\n```\s*$", s)
    if m:
        return m.group(1).strip()
    # Remove any lone backticks fencing
    if s.startswith("```"):
        s = s.lstrip("`")
    if s.endswith("```"):
        s = s.rstrip("`")
    return s.strip()


def _attempt_repair_json(raw: str) -> str:
    # Normalize quotes and remove fences
    t = raw.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    t = t.replace("```json", "").replace("```", "").strip()
    # Trim leading text before first { or [
    start_brace = t.find("{")
    start_bracket = t.find("[")
    starts = [x for x in (start_brace, start_bracket) if x != -1]
    if starts:
        t = t[min(starts):]
    # Trim trailing after last } or ]
    last_brace = t.rfind("}")
    last_bracket = t.rfind("]")
    last = max(last_brace, last_bracket)
    if last != -1:
        t = t[: last + 1]
    # Remove trailing commas before closing
    t = re.sub(r",\s*([}\]])", r"\1", t)

    def balance(s: str, open_ch: str, close_ch: str) -> str:
        count = 0
        for ch in s:
            if ch == open_ch:
                count += 1
            elif ch == close_ch and count > 0:
                count -= 1
        if count > 0:
            s += close_ch * count
        return s

    t = balance(t, "{", "}")
    t = balance(t, "[", "]")
    return t.strip()


def _parse_json_or_raise(text: str) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty model response; expected JSON.")
    s = _strip_code_fences(text)
    # Try direct JSON first
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return {"hypotheses": val}
        return val
    except Exception:
        pass
    # Extract JSON object or array substring
    m = _JSON_OBJ_RE.search(s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    m = _JSON_ARR_RE.search(s)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return {"hypotheses": arr}
        except Exception:
            pass
    # Attempt heuristic repair
    repaired = _attempt_repair_json(text)
    if repaired and repaired != s:
        try:
            val = json.loads(repaired)
            if isinstance(val, list):
                return {"hypotheses": val}
            return val
        except Exception:
            pass
    raise ValueError("Failed to parse JSON from model response.")


def _ensure_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if x is None:
        return []
    return [x]
