import os,json

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
