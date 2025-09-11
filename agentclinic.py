import argparse
import re, random, time, json, os
from openai import OpenAI
from typing import Optional


# OpenAI client management
_openai_client = None

def init_openai_client(api_key: Optional[str], base_url: Optional[str] = None):
    global _openai_client
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    _openai_client = OpenAI(**kwargs)


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        # Fallback to environment configuration
        _openai_client = OpenAI()
    return _openai_client


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


def query_model(model_str, prompt, system_prompt, tries=30, timeout=20.0, image_requested=False, scene=None, max_prompt_len=2**14, clip_prompt=False):
    """Query an OpenAI chat model using latest SDK.
    - model_str: exact model name (e.g., "gpt-4o-mini", "gpt-4o")
    - If image_requested is True and scene has image_url, send a multimodal message.
    """
    client = get_openai_client()
    for _ in range(tries):
        if clip_prompt:
            prompt = prompt[:max_prompt_len]
        try:
            if image_requested and scene is not None and getattr(scene, "image_url", None):
                user_content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"{scene.image_url}"}},
                ]
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            resp = client.chat.completions.create(
                model=model_str,
                messages=messages,
                temperature=0.05,
                max_tokens=200,
            )
            answer = resp.choices[0].message.content
            answer = re.sub(r"\s+", " ", answer)
            return answer
        except Exception:
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


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


def compare_results(diagnosis, correct_diagnosis, moderator_llm):
    prompts = load_prompts_json("moderator")
    user_prompt = prompts["user_prompt_template"].format(correct=correct_diagnosis, diag=diagnosis)
    system_prompt = prompts["system_prompt"]
    answer = query_model(moderator_llm, user_prompt, system_prompt)
    return answer.lower()


# =========================
# Main entry with config
# =========================

def main(config_path: str):
    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Initialize OpenAI client
    openai_cfg = cfg.get("openai", {})
    api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")
    base_url = openai_cfg.get("base_url")
    init_openai_client(api_key, base_url)

    # Run settings
    run_cfg = cfg.get("run", {})
    inf_type = run_cfg.get("inf_type", "llm")
    doctor_bias = run_cfg.get("doctor_bias", "None")
    patient_bias = run_cfg.get("patient_bias", "None")
    doctor_llm = run_cfg.get("doctor_llm", "gpt-4o-mini")
    patient_llm = run_cfg.get("patient_llm", "gpt-4o-mini")
    measurement_llm = run_cfg.get("measurement_llm", "gpt-4o-mini")
    moderator_llm = run_cfg.get("moderator_llm", "gpt-4o-mini")
    dataset = run_cfg.get("dataset", "MedQA")  # MedQA, MedQA_Ext, MIMICIV, NEJM, NEJM_Ext
    img_request = bool(run_cfg.get("doctor_image_request", False))
    num_scenarios = run_cfg.get("num_scenarios", None)
    total_inferences = int(run_cfg.get("total_inferences", 20))

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

    if num_scenarios is None:
        num_scenarios = scenario_loader.num_scenarios

    for _scenario_id in range(0, min(num_scenarios, scenario_loader.num_scenarios)):
        total_presents += 1
        pi_dialogue = str()
        # Initialize scenario
        scenario = scenario_loader.get_scenario(id=_scenario_id)
        # Initialize agents
        meas_agent = MeasurementAgent(
            scenario=scenario,
            backend_str=measurement_llm)
        patient_agent = PatientAgent(
            scenario=scenario,
            bias_present=patient_bias,
            backend_str=patient_llm)
        doctor_agent = DoctorAgent(
            scenario=scenario,
            bias_present=doctor_bias,
            backend_str=doctor_llm,
            max_infs=total_inferences,
            img_request=img_request)

        doctor_dialogue = ""
        for _inf_id in range(total_inferences):
            # Check for medical image request
            if dataset == "NEJM":
                if img_request:
                    imgs = "REQUEST IMAGES" in doctor_dialogue
                else:
                    imgs = True
            else:
                imgs = False
            # Check if final inference
            if _inf_id == total_inferences - 1:
                pi_dialogue += "This is the final question. Please provide a diagnosis.\n"
            # Obtain doctor dialogue (human or llm agent)
            if inf_type == "human_doctor":
                doctor_dialogue = input("\nQuestion for patient: ")
            else:
                doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue, image_requested=imgs)
            print("Doctor [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), doctor_dialogue)
            # Doctor has arrived at a diagnosis, check correctness
            if "DIAGNOSIS READY" in doctor_dialogue:
                correctness = compare_results(doctor_dialogue, scenario.diagnosis_information(), moderator_llm) == "yes"
                if correctness: total_correct += 1
                print("\nCorrect answer:", scenario.diagnosis_information())
                print("Scene {}, The diagnosis was ".format(_scenario_id), "CORRECT" if correctness else "INCORRECT", int((total_correct/total_presents)*100))
                break
            # Obtain medical exam from measurement reader
            if "REQUEST TEST" in doctor_dialogue:
                pi_dialogue = meas_agent.inference_measurement(doctor_dialogue,)
                print("Measurement [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
                patient_agent.add_hist(pi_dialogue)
            # Obtain response from patient
            else:
                if inf_type == "human_patient":
                    pi_dialogue = input("\nResponse to doctor: ")
                else:
                    pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
                print("Patient [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
                meas_agent.add_hist(pi_dialogue)
            # Prevent API timeouts
            time.sleep(1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Diagnosis Simulation')
    parser.add_argument('--config', type=str, default='agentclinic.config.json', help='Path to configuration JSON file')
    args = parser.parse_args()

    main(args.config)
