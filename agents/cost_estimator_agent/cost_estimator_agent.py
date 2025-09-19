from llm import query_model
from agents.agent_tools import load_prompts_json

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
