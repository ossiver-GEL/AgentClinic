from llm import query_model
from agent_tools import load_prompts_json

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
