import re
import time
from typing import Optional
from openai import OpenAI

# OpenAI client management (centralized here)
_openai_client = None

# Global config for Responses API kwargs and runtime controls
_llm_responses_kwargs = {}
_llm_runtime_cfg = {
    "tries": 30,
    "timeout": 20.0,
    "clip_prompt": False,
    "max_prompt_len": 2 ** 14,
}

def load_llm_config(path: str = "llm.config.json") -> dict:
        """Load LLM configuration JSON (containing 'openai' and 'llm' sections) and
        initialize global client + runtime settings.

        Parameters:
            path: Path to llm.config.json.
            required: If True, raise FileNotFoundError when missing; else return {}.

        Returns:
            A dictionary: { 'openai': {...}, 'llm': {...} }
        """
        import json, os
        if not os.path.exists(path):
            raise FileNotFoundError(f"LLM config file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        openai_cfg = cfg.get('openai', {})
        llm_cfg = cfg.get('llm', {})
        api_key = openai_cfg.get('api_key') or os.environ.get('OPENAI_API_KEY')
        base_url = openai_cfg.get('base_url')
        init_openai_client(api_key, base_url)
        responses_cfg = llm_cfg.get('responses')
        runtime_cfg = llm_cfg.get('runtime')
        set_llm_config(responses=responses_cfg, runtime=runtime_cfg)
        return cfg


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


def set_llm_config(responses: Optional[dict] = None, runtime: Optional[dict] = None):
    """Set global LLM configuration.
    - responses: will be merged into client.responses.create(**kwargs), except reserved keys (model, input)
    - runtime: controls retry/timeout/prompt clipping
    """
    global _llm_responses_kwargs, _llm_runtime_cfg
    if responses:
        # sanitize reserved keys
        sanitized = dict(responses)
        sanitized.pop("model", None)
        sanitized.pop("input", None)
        _llm_responses_kwargs = sanitized
    if runtime:
        for key in ("tries", "timeout", "clip_prompt", "max_prompt_len"):
            if key in runtime:
                _llm_runtime_cfg[key] = runtime[key]


def query_model(
    model_str,
    prompt,
    system_prompt,
    tries: int = 30,
    timeout: float = 20.0,
    image_requested: bool = False,
    scene=None,
    max_prompt_len: int = 2 ** 14,
    clip_prompt: bool = False,
):
    """Query an OpenAI model using the Responses API (unified for all supported models).
    - Unified usage: always call client.responses.create for GPT-4o/mini、GPT-5 系列等。
    - If image_requested is True and scene has image_url, send a multimodal message.
    """
    client = get_openai_client()

    # Allow global runtime cfg to override call-time defaults
    tries = _llm_runtime_cfg.get("tries", tries)
    timeout = _llm_runtime_cfg.get("timeout", timeout)
    clip_prompt = _llm_runtime_cfg.get("clip_prompt", clip_prompt)
    max_prompt_len = _llm_runtime_cfg.get("max_prompt_len", max_prompt_len)

    for _ in range(tries):
        if clip_prompt:
            prompt = prompt[:max_prompt_len]
        try:
            # Build input content for Responses API
            if image_requested and scene is not None and getattr(scene, "image_url", None):
                input_content = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"{getattr(scene, 'image_url', '')}"}},
                    ]},
                ]
            else:
                input_content = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]

            # Base kwargs for all models
            kwargs = dict(
                model=model_str,
                input=input_content,
                max_output_tokens=200,
            )
            # Merge globally configured Responses kwargs
            if _llm_responses_kwargs:
                # do not allow override of reserved keys
                reserved = {"model", "input"}
                is_gpt5 = model_str.startswith(("gpt-5", "gpt5"))
                for k, v in _llm_responses_kwargs.items():
                    if k in reserved:
                        continue
                    # guard GPT-5 specific knobs for other models
                    if (k == "reasoning" or k == "text") and not is_gpt5:
                        continue
                    kwargs[k] = v

            # Optional defaults for GPT-5 family if not provided by config
            if model_str.startswith(("gpt-5", "gpt5")):
                kwargs.setdefault("reasoning", {"effort": "minimal"})
                kwargs.setdefault("text", {"verbosity": "low"})

            resp = client.responses.create(**kwargs)

            # Prefer high-level helper, then fallback to manual parse if empty
            answer = getattr(resp, "output_text", None) or ""
            if not answer:
                try:
                    out = getattr(resp, "output", None) or []
                    for item in out:
                        if getattr(item, "type", None) == "message":
                            contents = getattr(item, "content", []) or []
                            for c in contents:
                                ctype = getattr(c, "type", None)
                                if ctype in ("output_text", "text"):
                                    txt = getattr(c, "text", None)
                                    if txt:
                                        answer = txt
                                        break
                            if answer:
                                break
                except Exception:
                    # Leave answer empty; retry loop will handle
                    pass

            answer = re.sub(r"\s+", " ", answer)
            if answer:
                return answer
            else:
                raise Exception("Empty response")
        except Exception:
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")