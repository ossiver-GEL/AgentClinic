from llm import query_model
from agents.agent_tools import load_prompts_json

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
