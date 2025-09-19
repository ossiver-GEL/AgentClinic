from llm import query_model
from agents.agent_tools import load_prompts_json

def compare_results(diagnosis, correct_diagnosis, moderator_llm):

    print(f"Comparing diagnosis '{diagnosis}' with correct '{correct_diagnosis}' using moderator LLM '{moderator_llm}'")

    prompts = load_prompts_json("moderator")
    user_prompt = prompts["user_prompt_template"].format(correct=correct_diagnosis, diag=diagnosis)
    system_prompt = prompts["system_prompt"]
    answer = query_model(moderator_llm, user_prompt, system_prompt)
    return answer.lower()