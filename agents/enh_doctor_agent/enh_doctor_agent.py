from llm import query_model
from agents.agent_tools import load_prompts_json


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
