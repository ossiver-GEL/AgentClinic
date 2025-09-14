import argparse
import re, random, time, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Optional
from datetime import datetime

from llm import init_openai_client, get_openai_client, query_model, set_llm_config, load_llm_config
import json as _json_mod


# Prompt loading with caching
PROMPT_CACHE = {}

def load_prompts_json(name: str) -> dict:
    global PROMPT_CACHE
    if name in PROMPT_CACHE:
        return PROMPT_CACHE[name]
    base_dir = os.path.join(os.path.dirname(__file__), "prompts")
    path = os.path.join(base_dir, f"{name}.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    PROMPT_CACHE[name] = data
    return data

class ScenarioMedQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQA:
    def __init__(self, file_path: Optional[str] = None) -> None:
        path = file_path or "agentclinic_medqa.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]
        


class ScenarioMedQAExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQAExtended:
    def __init__(self, file_path: Optional[str] = None) -> None:
        path = file_path or "agentclinic_medqa_extended.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQAExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]
        


class ScenarioMIMICIVQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMIMICIV:
    def __init__(self, file_path: Optional[str] = None) -> None:
        path = file_path or "agentclinic_mimiciv.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMIMICIVQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioNEJMExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict 
        self.question = scenario_dict["question"] 
        self.image_url = scenario_dict["image_url"] 
        self.diagnosis = [_sd["text"] 
            for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"
    
    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams
    
    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJMExtended:
    def __init__(self, file_path: Optional[str] = None) -> None:
        path = file_path or "agentclinic_nejm_extended.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJMExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioNEJM:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict 
        self.question = scenario_dict["question"] 
        self.image_url = scenario_dict["image_url"] 
        self.diagnosis = [_sd["text"] 
            for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"
    
    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams
    
    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJM:
    def __init__(self, file_path: Optional[str] = None) -> None:
        path = file_path or "agentclinic_nejm.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJM(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class PatientAgent:
    def __init__(self, scenario, backend_str="gpt-4o-mini", bias_present=None) -> None:
        # disease of patient, or "correct answer"
        self.disease = ""
        # symptoms that patient presents
        self.symptoms = ""
        # conversation history between doctor and patient
        self.agent_hist = ""
        # language model backend for patient agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # sample initial question from dataset
        self.scenario = scenario
        self.reset()
        self.pipe = None

        self.biases = ["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        """ 
        ================
        Cognitive biases 
        ================
        """
        if self.bias_present is None:
            return ""
        prompts = load_prompts_json("patient").get("biases", {})
        if self.bias_present in prompts:
            return prompts[self.bias_present]
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
            return ""

    def inference_patient(self, question) -> str:
        prompts = load_prompts_json("patient")
        user_prompt = prompts["user_prompt_template"].format(agent_hist=self.agent_hist, question=question)
        answer = query_model(self.backend, user_prompt, self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        prompts = load_prompts_json("patient")
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = prompts["system_base"]
        symptoms = prompts["system_symptoms_suffix"].format(self.symptoms)
        return base + bias_prompt + symptoms
    
    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"


class DoctorAgent:
    def __init__(self, scenario, backend_str="gpt-4o-mini", max_infs=20, bias_present=None, img_request=False) -> None:
        # number of inference calls to the doctor
        self.infs = 0
        # maximum number of inference calls to the doctor
        self.MAX_INFS = max_infs
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for doctor
        self.presentation = ""
        # language model backend for doctor agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.reset()
        self.pipe = None
        self.img_request = img_request
        self.biases = ["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        """ 
        ================
        Cognitive biases 
        ================
        """
        if self.bias_present is None:
            return ""
        prompts = load_prompts_json("doctor").get("biases", {})
        if self.bias_present in prompts:
            return prompts[self.bias_present]
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
            return ""

    def inference_doctor(self, question, image_requested=False) -> str:
        answer = str()
        if self.infs >= self.MAX_INFS: return "Maximum inferences reached"
        prompts = load_prompts_json("doctor")
        user_prompt = prompts["user_prompt_template"].format(agent_hist=self.agent_hist, question=question)
        answer = query_model(self.backend, user_prompt, self.system_prompt(), image_requested=image_requested, scene=self.scenario)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.infs += 1
        return answer

    def system_prompt(self) -> str:
        prompts = load_prompts_json("doctor")
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = prompts["system_base"].format(self.MAX_INFS, self.infs)
        base_with_images = base + (prompts.get("system_images_suffix", "") if self.img_request else "")
        presentation = prompts["system_presentation_suffix"].format(self.presentation)
        return base_with_images + bias_prompt + presentation

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()


class MeasurementAgent:
    def __init__(self, scenario, backend_str="gpt-4o-mini") -> None:
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for measurement 
        self.presentation = ""
        # language model backend for measurement agent
        self.backend = backend_str
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.pipe = None
        self.reset()

    def inference_measurement(self, question) -> str:
        prompts = load_prompts_json("measurement")
        user_prompt = prompts["user_prompt_template"].format(agent_hist=self.agent_hist, question=question)
        answer = query_model(self.backend, user_prompt, self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        prompts = load_prompts_json("measurement")
        base = prompts["system_base"]
        presentation = prompts["system_presentation_suffix"].format(self.information)
        return base + presentation
    
    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()


class CostEstimatorAgent:
    def __init__(self, scenario, backend_str="gpt-4o-mini") -> None:
        self.agent_hist = ""
        self.backend = backend_str
        self.scenario = scenario
        self.pipe = None
        self.reset()

    def inference_cost(self, question: str, resource_level: str = "normal") -> str:
        prompts = load_prompts_json("measurement_cost")
        user_prompt = prompts["user_prompt_template"].format(
            agent_hist=self.agent_hist,
            question=question,
            resource_level=resource_level,
            scenario_info=self.information,
        )
        answer = query_model(self.backend, user_prompt, self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        prompts = load_prompts_json("measurement_cost")
        base = prompts["system_base"]
        presentation = prompts.get("system_presentation_suffix", "").format(self.information)
        return base + presentation

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()


def _extract_requested_test(dialogue: str) -> str:
    try:
        m = re.search(r"REQUEST\s+TEST\s*:\s*([^\n\r]+)", dialogue, flags=re.IGNORECASE)
        return m.group(1).strip() if m else ""
    except Exception:
        return ""


def _safe_parse_json(text: str):
    try:
        return _json_mod.loads(text)
    except Exception:
        pass
    try:
        start = text.find("{"); end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return _json_mod.loads(text[start:end+1])
    except Exception:
        pass
    try:
        start = text.find("["); end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return _json_mod.loads(text[start:end+1])
    except Exception:
        pass
    return None


def compare_results(diagnosis, correct_diagnosis, moderator_llm):
    prompts = load_prompts_json("moderator")
    user_prompt = prompts["user_prompt_template"].format(correct=correct_diagnosis, diag=diagnosis)
    system_prompt = prompts["system_prompt"]
    answer = query_model(moderator_llm, user_prompt, system_prompt)
    return answer.lower()


# =========================
# Main entry with config
# =========================

def main(config_path: str, llm_config_path: str, workers: Optional[int] = None):
    # Load main (run + datasets) configuration
    with open(config_path, "r", encoding="utf-8") as f:
        main_cfg = json.load(f)

    # Load & apply LLM configuration centrally
    llm_file_cfg = load_llm_config()
    openai_cfg = llm_file_cfg.get("openai", {})
    llm_cfg = llm_file_cfg.get("llm", {})
    responses_cfg = llm_cfg.get("responses")
    runtime_cfg = llm_cfg.get("runtime")

    # Construct merged cfg for downstream (do not reintroduce deprecated structure)
    cfg = {
        **main_cfg,
        "openai": openai_cfg,
        "llm": llm_cfg,
        "_meta": {
            "main_config": config_path,
            "llm_config": llm_config_path
        }
    }

    # Run settings (including workers which can come from config if CLI not set)
    run_cfg = cfg.get("run", {})
    inf_type = run_cfg.get("inf_type", "llm")
    doctor_bias = run_cfg.get("doctor_bias", "None")
    patient_bias = run_cfg.get("patient_bias", "None")
    doctor_llm = run_cfg.get("doctor_llm", "gpt-4o-mini")
    patient_llm = run_cfg.get("patient_llm", "gpt-4o-mini")
    measurement_llm = run_cfg.get("measurement_llm", "gpt-4o-mini")
    moderator_llm = run_cfg.get("moderator_llm", "gpt-4o-mini")
    measurement_cost_llm = run_cfg.get("measurement_cost_llm", measurement_llm)
    resource_level = run_cfg.get("resource_level", "normal")
    dataset = run_cfg.get("dataset", "MedQA")  # MedQA, MedQA_Ext, MIMICIV, NEJM, NEJM_Ext
    img_request = bool(run_cfg.get("doctor_image_request", False))
    num_scenarios = run_cfg.get("num_scenarios", None)
    total_inferences = int(run_cfg.get("total_inferences", 20))
    # Determine workers precedence: CLI arg (if not None) overrides config; else fallback to config or 1
    cfg_workers = int(run_cfg.get("workers", 1))
    effective_workers = int(workers) if workers is not None else cfg_workers

    # Initialize runs output directory and artifacts
    output_root = run_cfg.get("output_dir", "runs")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(output_root, f"{dataset}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    # Persist the effective config for this run
    try:
        with open(os.path.join(out_dir, "config.used.json"), "w", encoding="utf-8") as f_out:
            json.dump(cfg, f_out, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Warning: failed to write config.used.json:", e)
    scenario_jsonl_path = os.path.join(out_dir, "scenarios.jsonl")
    run_start_time = time.time()

    # Dataset paths (optional overrides)
    ds_paths = cfg.get("datasets", {})
    ds_file = ds_paths.get(dataset)

    # Load scenario loader
    if dataset == "MedQA":
        scenario_loader = ScenarioLoaderMedQA(file_path=ds_file)
    elif dataset == "MedQA_Ext":
        scenario_loader = ScenarioLoaderMedQAExtended(file_path=ds_file)
    elif dataset == "NEJM":
        scenario_loader = ScenarioLoaderNEJM(file_path=ds_file)
    elif dataset == "NEJM_Ext":
        scenario_loader = ScenarioLoaderNEJMExtended(file_path=ds_file)
    elif dataset == "MIMICIV":
        scenario_loader = ScenarioLoaderMIMICIV(file_path=ds_file)
    else:
        raise Exception("Dataset {} does not exist".format(str(dataset)))

    total_correct = 0
    total_presents = 0
    # Aggregates for test economics across the entire run
    total_tests_agg = 0
    total_cost_agg_usd = 0.0

    if num_scenarios is None:
        num_scenarios = scenario_loader.num_scenarios

    # Parallel worker for a single scenario
    lock = threading.Lock()

    def run_scenario(scenario_id: int):
        nonlocal total_correct, total_tests_agg, total_cost_agg_usd
        scenario = scenario_loader.get_scenario(id=scenario_id)
        meas_agent = MeasurementAgent(scenario=scenario, backend_str=measurement_llm)
        patient_agent = PatientAgent(scenario=scenario, bias_present=patient_bias, backend_str=patient_llm)
        doctor_agent = DoctorAgent(scenario=scenario, bias_present=doctor_bias, backend_str=doctor_llm, max_infs=total_inferences, img_request=img_request)
        cost_agent = CostEstimatorAgent(scenario=scenario, backend_str=measurement_cost_llm)
        pi_dialogue = ""
        doctor_dialogue = ""
        diagnosed = False
        test_economics_log = []
        scenario_cost_total = 0.0
        scenario_wait_total_min = 0.0
        scenario_duration_total_min = 0.0
        for _inf_id in range(total_inferences):
            if dataset == "NEJM":
                imgs = ("REQUEST IMAGES" in doctor_dialogue) if img_request else True
            else:
                imgs = False
            if _inf_id == total_inferences - 1:
                pi_dialogue += "This is the final question. Please provide a diagnosis.\n"
            if inf_type == "human_doctor":
                doctor_dialogue = input(f"\n[Scene {scenario_id}] Question for patient: ")
            else:
                doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue, image_requested=imgs)
            if "DIAGNOSIS READY" in doctor_dialogue:
                correctness = compare_results(doctor_dialogue, scenario.diagnosis_information(), moderator_llm) == "yes"
                with lock:
                    if correctness:
                        total_correct += 1
                record = {
                    "scenario_id": scenario_id,
                    "dataset": dataset,
                    "status": "diagnosed",
                    "num_turns": _inf_id + 1,
                    "correct": bool(correctness),
                    "ground_truth": scenario.diagnosis_information(),
                    "diagnosis_text": doctor_dialogue,
                    "doctor_llm": doctor_llm,
                    "patient_llm": patient_llm,
                    "measurement_llm": measurement_llm,
                    "moderator_llm": moderator_llm,
                    "max_infs": total_inferences,
                    "image_request_enabled": img_request,
                    "doctor_hist": doctor_agent.agent_hist,
                    "patient_hist": patient_agent.agent_hist,
                    "measurement_hist": meas_agent.agent_hist,
                    "llm_responses_config": responses_cfg,
                    "llm_runtime_config": runtime_cfg,
                    "resource_level": resource_level,
                    "test_economics_log": test_economics_log,
                    "test_economics_totals": {
                        "total_estimated_cost_usd": scenario_cost_total,
                        "total_expected_wait_time_minutes": scenario_wait_total_min,
                        "total_expected_duration_minutes": scenario_duration_total_min,
                    },
                }
                with lock:
                    with open(scenario_jsonl_path, "a", encoding="utf-8") as f_out:
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                diagnosed = True
                break
            if "REQUEST TEST" in doctor_dialogue:
                pi_dialogue = meas_agent.inference_measurement(doctor_dialogue)
                patient_agent.add_hist(pi_dialogue)
                # Estimate cost/time for requested test (non-blocking best-effort)
                try:
                    req_name = _extract_requested_test(doctor_dialogue)
                    cost_resp = cost_agent.inference_cost(doctor_dialogue, resource_level=resource_level)
                    payload = _safe_parse_json(cost_resp) or {}
                    entries = []
                    if isinstance(payload, dict) and isinstance(payload.get("tests"), list):
                        entries = payload["tests"]
                    elif isinstance(payload, list):
                        entries = payload
                    for ent in entries:
                        try:
                            name = ent.get("test_name") or req_name
                            curr = ent.get("estimate_currency") or "USD"
                            cost = float(ent.get("estimate_cost") or 0)
                            wait_min = float(ent.get("expected_wait_time_minutes") or 0)
                            dur_min = float(ent.get("expected_duration_minutes") or 0)
                            assumptions = ent.get("assumptions") or ""
                            test_economics_log.append({
                                "test_name": name,
                                "estimate_currency": curr,
                                "estimate_cost": cost,
                                "expected_wait_time_minutes": wait_min,
                                "expected_duration_minutes": dur_min,
                                "assumptions": assumptions,
                                "resource_level": resource_level,
                            })
                            scenario_cost_total += cost
                            scenario_wait_total_min += wait_min
                            scenario_duration_total_min += dur_min
                            with lock:
                                total_tests_agg += 1
                                if curr == "USD":
                                    total_cost_agg_usd += cost
                        except Exception:
                            continue
                except Exception:
                    pass
            else:
                if inf_type == "human_patient":
                    pi_dialogue = input(f"\n[Scene {scenario_id}] Response to doctor: ")
                else:
                    pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
                meas_agent.add_hist(pi_dialogue)
            time.sleep(1.0)
        if not diagnosed:
            record = {
                "scenario_id": scenario_id,
                "dataset": dataset,
                "status": "max_infs_reached",
                "num_turns": total_inferences,
                "correct": None,
                "ground_truth": scenario.diagnosis_information(),
                "final_doctor_dialogue": doctor_dialogue,
                "doctor_llm": doctor_llm,
                "patient_llm": patient_llm,
                "measurement_llm": measurement_llm,
                "moderator_llm": moderator_llm,
                "max_infs": total_inferences,
                "image_request_enabled": img_request,
                "doctor_hist": doctor_agent.agent_hist,
                "patient_hist": patient_agent.agent_hist,
                "measurement_hist": meas_agent.agent_hist,
                "llm_responses_config": responses_cfg,
                "llm_runtime_config": runtime_cfg,
                "resource_level": resource_level,
                "test_economics_log": test_economics_log,
                "test_economics_totals": {
                    "total_estimated_cost_usd": scenario_cost_total,
                    "total_expected_wait_time_minutes": scenario_wait_total_min,
                    "total_expected_duration_minutes": scenario_duration_total_min,
                },
            }
            with lock:
                with open(scenario_jsonl_path, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    scenario_range = list(range(0, min(num_scenarios, scenario_loader.num_scenarios)))
    total_presents = len(scenario_range)
    if effective_workers <= 1 or inf_type.startswith("human_"):
        for sid in scenario_range:
            run_scenario(sid)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(run_scenario, sid): sid for sid in scenario_range}
            for fut in as_completed(futures):
                _ = futures[fut]

    # Write summary of the run
    try:
        total_time = time.time() - run_start_time
        summary = {
            "dataset": dataset,
            "doctor_llm": doctor_llm,
            "patient_llm": patient_llm,
            "measurement_llm": measurement_llm,
            "moderator_llm": moderator_llm,
            "num_scenarios": min(num_scenarios, scenario_loader.num_scenarios),
            "total_presented": total_presents,
            "total_correct": total_correct,
            "accuracy": (total_correct/total_presents) if total_presents > 0 else None,
            "total_inferences": total_inferences,
            "duration_seconds": total_time,
            "timestamp": ts,
            "output_dir": out_dir,
            "scenario_file": os.path.basename(scenario_jsonl_path),
            "llm_responses_config": responses_cfg,
            "llm_runtime_config": runtime_cfg,
            "resource_level": resource_level,
            "total_tests": total_tests_agg,
            "total_estimated_cost_usd": total_cost_agg_usd,
        }
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f_out:
            json.dump(summary, f_out, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Warning: failed to write summary.json:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Diagnosis Simulation')
    parser.add_argument('--config', type=str, default='agentclinic.config.json', help='Path to main configuration JSON file (run + datasets)')
    parser.add_argument('--llm-config', type=str, default='llm.config.json', help='Path to LLM configuration JSON file (openai + llm)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel scenario workers (ignored for human_* modes). Overrides config run.workers when provided.')
    args = parser.parse_args()

    main(args.config, args.llm_config, args.workers)
