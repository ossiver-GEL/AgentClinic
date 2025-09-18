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
    # retry_delay: sleep seconds between retries on failure
    "timeout": 20.0,
    # request_timeout: per-request HTTP timeout (seconds) to avoid hangs
    "request_timeout": 30.0,
    "clip_prompt": False,
    "max_prompt_len": 2 ** 14,
    # llm_api: 'responses' | 'chat' | 'auto' (auto tries responses then falls back to chat)
    "llm_api": "auto",
}

def load_llm_config(path: str = "llm.config.json") -> dict:
        """Load LLM configuration JSON (containing 'openai' and 'llm' sections) and
        initialize global client + runtime settings.

        Parameters:
            path: Path to llm.config.json.

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
        for key in ("tries", "timeout", "request_timeout", "clip_prompt", "max_prompt_len", "llm_api"):
            if key in runtime:
                _llm_runtime_cfg[key] = runtime[key]


def _build_user_content_for_mm(prompt: str, image_url: Optional[str]):
    """Builds the multimodal content array for a user message, used by both APIs.
    For chat.completions this array goes into the user message's content.
    For responses API, the same structure is valid in `input`.
    """
    if image_url:
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    return prompt


def _merge_kwargs_for_chat(base: dict, model_str: str) -> dict:
    """Map configured responses-style kwargs into chat.completions equivalents.
    - max_output_tokens -> max_tokens
    - ignore responses-only keys like reasoning/text
    - pass-through common sampling params if present
    """
    kwargs = dict(base) if base else {}
    # Remove responses-only or reserved keys
    for k in ("model", "input"):
        kwargs.pop(k, None)

    # Map token limit
    if "max_output_tokens" in kwargs and "max_tokens" not in kwargs:
        kwargs["max_tokens"] = kwargs.pop("max_output_tokens")

    # Remove Responses-API specific fields that chat API doesn't accept
    kwargs.pop("reasoning", None)
    kwargs.pop("text", None)

    # Keep known chat params if provided in config: temperature, top_p, n, stop, presence_penalty, frequency_penalty, seed, logit_bias, response_format, tools, tool_choice, parallel_tool_calls
    # (No special handling needed; they pass through if present and supported by backend.)

    # Return sanitized kwargs for chat.completions
    return kwargs


def _extract_text_from_chat(resp) -> str:
    try:
        choice0 = resp.choices[0]
        content = getattr(choice0.message, "content", None)
        if isinstance(content, str):
            return content
        # Some SDKs might return a list of content parts; join text parts if so
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                    t = part.get("text")
                    if t:
                        texts.append(t)
            return " ".join(texts)
    except Exception:
        pass
    return ""


def _extract_text_from_responses(resp) -> str:
    # Prefer high-level helper, then fallback to manual parse if empty
    answer = getattr(resp, "output_text", None) or ""
    if answer:
        return answer
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
                            return txt
    except Exception:
        pass
    return ""


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
    """Query an OpenAI model.
    Default: Use Responses API where available, but supports Chat Completions as a full-featured fallback.
    - If image_requested is True and scene has image_url, send a multimodal message.
    - Runtime switch: set runtime.llm_api to 'responses' | 'chat' | 'auto'.
    """
    client = get_openai_client()

    # Allow global runtime cfg to override call-time defaults
    tries = _llm_runtime_cfg.get("tries", tries)
    retry_delay = _llm_runtime_cfg.get("timeout", timeout)
    request_timeout = _llm_runtime_cfg.get("request_timeout", 30.0)
    clip_prompt = _llm_runtime_cfg.get("clip_prompt", clip_prompt)
    max_prompt_len = _llm_runtime_cfg.get("max_prompt_len", max_prompt_len)
    llm_api = _llm_runtime_cfg.get("llm_api", "auto")

    for _ in range(tries):
        if clip_prompt:
            prompt = prompt[:max_prompt_len]
        try:
            # Build content/messages once
            image_url = getattr(scene, "image_url", None) if (image_requested and scene is not None) else None
            user_content = _build_user_content_for_mm(prompt, image_url)

            # Apply per-request timeout to avoid indefinite hangs
            client_req = client
            try:
                client_req = client.with_options(timeout=request_timeout)
            except Exception:
                pass

            def call_responses_api():
                # Base kwargs for Responses API
                r_kwargs = dict(
                    model=model_str,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_output_tokens=200,
                )
                if _llm_responses_kwargs:
                    reserved = {"model", "input"}
                    is_gpt5 = model_str.startswith(("gpt-5", "gpt5"))
                    for k, v in _llm_responses_kwargs.items():
                        if k in reserved:
                            continue
                        if (k == "reasoning" or k == "text") and not is_gpt5:
                            continue
                        r_kwargs[k] = v
                if model_str.startswith(("gpt-5", "gpt5")):
                    r_kwargs.setdefault("reasoning", {"effort": "minimal"})
                    r_kwargs.setdefault("text", {"verbosity": "low"})
                return client_req.responses.create(**r_kwargs)

            def call_chat_api():
                # Prepare messages for Chat Completions
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
                # Map config to chat kwargs
                chat_cfg = _merge_kwargs_for_chat(_llm_responses_kwargs, model_str)
                c_kwargs = dict(model=model_str, messages=messages)
                # Provide a safe default tokens if not configured
                c_kwargs.setdefault("max_tokens", chat_cfg.pop("max_tokens", 200))
                # Merge remaining config
                c_kwargs.update(chat_cfg)
                return client_req.chat.completions.create(**c_kwargs)

            resp = None
            use_responses_first = llm_api in ("responses", "auto")
            use_chat_first = llm_api == "chat"
            error_primary = None

            if use_responses_first and not use_chat_first:
                try:
                    resp = call_responses_api()
                    answer = _extract_text_from_responses(resp)
                except Exception as e:
                    error_primary = e
                    resp = None
            if resp is None:
                try:
                    resp = call_chat_api()
                    answer = _extract_text_from_chat(resp)
                except Exception:
                    # If chat also fails, and we preferred chat, try responses as last resort
                    if llm_api == "chat":
                        try:
                            resp = call_responses_api()
                            answer = _extract_text_from_responses(resp)
                        except Exception:
                            raise
                    else:
                        # If responses failed first and chat failed too, re-raise the first error if available
                        if error_primary is not None:
                            raise error_primary
                        raise

            answer = re.sub(r"\s+", " ", answer)
            if answer:
                return answer
            else:
                raise Exception("Empty response")
        except Exception:
            time.sleep(retry_delay)
            continue
    raise Exception("Max retries: timeout")

if __name__ == "__main__":
    # Simple test
    cfg = load_llm_config(path="localllm.config.json")
    print("LLM config loaded:", cfg)
    test_prompt = "List 3 typical symptoms of a cold."
    test_system = "You are a medical expert."
    response = query_model("openbiollm8b", test_prompt, test_system)
    print("Response:", response)

'''
source .venvs/vllm/bin/activate
'''

'''
vllm serve openbiollm8b   --chat-template ~/llama3_chat_template.jinja   --dtype auto   --quantization bitsandbytes   --max-model-len 8192   --gpu-memory-utilization 0.90   --swap-space 8   --api-key sk-local-anything --enforce-eager
'''

'''
litellm --config litellm.config.yaml --host 0.0.0.0 --port 4000
'''