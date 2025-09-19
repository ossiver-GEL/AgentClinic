import os, json

# Prompt loading with caching
PROMPT_CACHE = {}

_NAME_TO_PATH = {
    # name -> (subfolder, filename)
    "doctor": ("doctor_agent", "doctor.json"),
    "enhanced_doctor": ("enh_doctor_agent", "enh_doctor.json"),
    "patient": ("patient_agent", "patient.json"),
    "measurement": ("meas_agent", "measurement.json"),
    "measurement_cost": ("cost_estimator_agent", "measurement_cost.json"),
    "moderator": ("moderator_agent", "moderator.json"),
}


def _resolve_prompt_path(name: str) -> str:
    base_dir = os.path.dirname(__file__)
    # Preferred: per-agent subfolder mapping
    if name in _NAME_TO_PATH:
        sub, fname = _NAME_TO_PATH[name]
        path = os.path.join(base_dir, sub, fname)
        if os.path.exists(path):
            return path
    # Legacy fallback: agents/prompts/{name}.json
    legacy = os.path.join(base_dir, "prompts", f"{name}.json")
    if os.path.exists(legacy):
        return legacy
    # Try a generic search in immediate subfolders for {name}.json
    for entry in os.listdir(base_dir):
        p = os.path.join(base_dir, entry, f"{name}.json")
        if os.path.exists(p):
            return p
    # Not found
    return ""


def load_prompts_json(name: str) -> dict:
    global PROMPT_CACHE
    if name in PROMPT_CACHE:
        return PROMPT_CACHE[name]
    path = _resolve_prompt_path(name)
    if not path:
        raise FileNotFoundError(
            f"Prompt file for '{name}' not found under agents/*/ or agents/prompts/."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    PROMPT_CACHE[name] = data
    return data
