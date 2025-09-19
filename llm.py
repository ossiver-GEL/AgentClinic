import re
import time
from typing import Optional, Tuple, Dict, Any
from openai import OpenAI


# -------------------------
# 全局：多端点/多配置路由容器
# -------------------------
_client_pool: Dict[str, OpenAI] = {}
_endpoint_cfgs: Dict[str, Dict[str, Any]] = {}  # endpoint_id -> {"responses": {...}, "runtime": {...}}
_model_to_endpoint: Dict[str, str] = {}
_default_endpoint: Optional[str] = None

# 默认运行时兜底（端点未设置时使用这些默认值）
_DEFAULT_RUNTIME = {
    "tries": 30,
    # retry_delay: 失败重试之间的睡眠秒数（与 timeout 区分开）
    "retry_delay": 1.0,
    # timeout: 仅用于外层容错（可与 request_timeout 分离使用）
    "timeout": 20.0,
    # request_timeout: HTTP 请求层的超时，避免挂死
    "request_timeout": 30.0,
    "clip_prompt": False,
    "max_prompt_len": 2 ** 14,
    # llm_api: 'responses' | 'chat' | 'auto'
    "llm_api": "auto",
}


# -------------------------
# 配置加载
# -------------------------
def load_llm_config(path: str = "llm.config.json") -> dict:
    """加载新格式 LLM 配置（含多端点、模型名路由），初始化客户端池与端点配置。

    必须包含:
      - endpoints: { endpoint_id: { base_url, api_key | api_key_env, llm:{responses, runtime} } }
      - model_map: { model_name: endpoint_id }
    可选:
      - default_endpoint: endpoint_id
    """
    import json, os

    if not os.path.exists(path):
        raise FileNotFoundError(f"LLM config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    endpoints = cfg.get("endpoints") or {}
    model_map = cfg.get("model_map") or {}
    default_endpoint = cfg.get("default_endpoint")

    if not endpoints or not isinstance(endpoints, dict):
        raise ValueError("Invalid config: 'endpoints' must be a non-empty object.")
    if not model_map or not isinstance(model_map, dict):
        raise ValueError("Invalid config: 'model_map' must be a non-empty object.")

    # 清空全局路由状态
    _client_pool.clear()
    _endpoint_cfgs.clear()
    _model_to_endpoint.clear()

    # 初始化端点与对应配置
    for eid, e in endpoints.items():
        if not isinstance(e, dict):
            raise ValueError(f"Invalid endpoint definition for '{eid}'.")
        base_url = e.get("base_url")
        if not base_url:
            raise ValueError(f"Endpoint '{eid}' missing 'base_url'.")

        api_key = e.get("api_key")
        api_key_env = e.get("api_key_env")
        if api_key is None and api_key_env:
            api_key = os.environ.get(api_key_env)

        # 注意：某些本地/自建网关可能不需要 API Key，允许为 None
        client_kwargs = {"base_url": base_url}
        client_kwargs["api_key"] = api_key

        _client_pool[eid] = OpenAI(**client_kwargs)

        llm = e.get("llm", {}) or {}
        responses_cfg = llm.get("responses", {}) or {}
        runtime_cfg = llm.get("runtime", {}) or {}
        _endpoint_cfgs[eid] = {
            "responses": dict(responses_cfg),
            "runtime": dict(runtime_cfg),
        }

    # 记录模型路由与默认端点
    _model_to_endpoint.update(model_map)
    global _default_endpoint
    _default_endpoint = default_endpoint

    return cfg


# -------------------------
# 路由与参数整备
# -------------------------
def _resolve_endpoint_for_model(model_str: str) -> Tuple[OpenAI, Dict[str, Any], Dict[str, Any], str]:
    """根据模型名解析端点、responses 配置、runtime 配置和端点 ID。"""
    eid = _model_to_endpoint.get(model_str) or _default_endpoint
    if not eid:
        raise ValueError(
            f"Model '{model_str}' not mapped and no 'default_endpoint' provided. "
            f"Available models in model_map: {sorted(_model_to_endpoint.keys())}"
        )
    client = _client_pool.get(eid)
    if client is None:
        raise ValueError(f"Endpoint '{eid}' has no initialized client. Check your config.")

    endpoint_cfg = _endpoint_cfgs.get(eid, {})
    responses_cfg = endpoint_cfg.get("responses", {}) or {}
    runtime_cfg = {**_DEFAULT_RUNTIME, **(endpoint_cfg.get("runtime", {}) or {})}
    return client, dict(responses_cfg), runtime_cfg, eid


def _build_user_content_for_mm(prompt: str, image_url: Optional[str]):
    """构造多模态内容（Responses 与 Chat 共用）。"""
    if image_url:
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    return prompt


def _merge_kwargs_for_chat(base: dict, model_str: str) -> dict:
    """将 Responses 风格参数映射/过滤为 Chat API 可接受的 kwargs。"""
    kwargs = dict(base) if base else {}
    # 移除 Responses 的保留字段
    for k in ("model", "input"):
        kwargs.pop(k, None)
    # token 上限映射
    if "max_output_tokens" in kwargs and "max_tokens" not in kwargs:
        kwargs["max_tokens"] = kwargs.pop("max_output_tokens")
    # Chat 不支持的字段剔除
    kwargs.pop("reasoning", None)
    kwargs.pop("text", None)
    return kwargs


def _extract_text_from_chat(resp) -> str:
    try:
        choice0 = resp.choices[0]
        content = getattr(choice0.message, "content", None)
        if isinstance(content, str):
            return content
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


# -------------------------
# 统一查询入口
# -------------------------
def query_model(
    model_str: str,
    prompt: str,
    system_prompt: str,
    tries: Optional[int] = None,
    timeout: Optional[float] = None,
    image_requested: bool = False,
    scene=None,
    max_prompt_len: Optional[int] = None,
    clip_prompt: Optional[bool] = None,
) -> str:
    """按模型名路由到对应端点，默认优先 Responses API，失败时可回退 Chat（由 runtime.llm_api 决定）。"""
    print(f"[LLM] model='{model_str}' prompt_len={len(prompt)}")

    client, responses_cfg, runtime_cfg, eid = _resolve_endpoint_for_model(model_str)

    # 让调用方的显式参数覆盖端点配置
    final_tries = tries if tries is not None else runtime_cfg.get("tries", _DEFAULT_RUNTIME["tries"])
    final_timeout = timeout if timeout is not None else runtime_cfg.get("timeout", _DEFAULT_RUNTIME["timeout"])
    final_retry_delay = runtime_cfg.get("retry_delay", _DEFAULT_RUNTIME["retry_delay"])
    request_timeout = runtime_cfg.get("request_timeout", _DEFAULT_RUNTIME["request_timeout"])
    final_clip = clip_prompt if clip_prompt is not None else runtime_cfg.get("clip_prompt", _DEFAULT_RUNTIME["clip_prompt"])
    final_max_prompt_len = max_prompt_len if max_prompt_len is not None else runtime_cfg.get("max_prompt_len", _DEFAULT_RUNTIME["max_prompt_len"])
    llm_api = runtime_cfg.get("llm_api", _DEFAULT_RUNTIME["llm_api"])

    # 构造多模态内容
    image_url = getattr(scene, "image_url", None) if (image_requested and scene is not None) else None
    user_content = _build_user_content_for_mm(prompt, image_url)

    # 按需裁剪提示
    if final_clip and isinstance(prompt, str):
        prompt = prompt[:final_max_prompt_len]

    # 保持请求级别超时
    client_req = client
    try:
        client_req = client.with_options(timeout=request_timeout)
    except Exception:
        pass

    def call_responses_api():
        r_kwargs = dict(
            model=model_str,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_output_tokens=responses_cfg.get("max_output_tokens", 200),
        )
        # 合并端点级 responses 配置，剔除保留字段
        for k, v in responses_cfg.items():
            if k in ("model", "input"):
                continue
            # OpenAI gpt-5 系列支持 reasoning/text，其他端点/模型若不支持可由后端返回 4xx/5xx，我们在外层处理重试/回退
            is_gpt5 = model_str.startswith(("gpt-5", "gpt5"))
            if (k in ("reasoning", "text")) and not is_gpt5:
                continue
            if k not in r_kwargs:
                r_kwargs[k] = v

        # 针对 gpt-5 系列默认开启轻量推理，尽量不影响其他模型
        if model_str.startswith(("gpt-5", "gpt5")):
            r_kwargs.setdefault("reasoning", {"effort": "minimal"})
            r_kwargs.setdefault("text", {"verbosity": "low"})

        return client_req.responses.create(**r_kwargs)

    def call_chat_api():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        chat_cfg = _merge_kwargs_for_chat(responses_cfg, model_str)
        c_kwargs = dict(model=model_str, messages=messages)
        c_kwargs.setdefault("max_tokens", chat_cfg.pop("max_tokens", 200))
        c_kwargs.update(chat_cfg)
        return client_req.chat.completions.create(**c_kwargs)

    # 调度策略
    use_responses_first = llm_api in ("responses", "auto")
    use_chat_first = llm_api == "chat"

    last_error = None
    for _ in range(final_tries):
        try:
            if use_responses_first and not use_chat_first:
                try:
                    resp = call_responses_api()
                    ans = _extract_text_from_responses(resp)
                    if not ans:
                        raise RuntimeError("Empty response text from Responses API")
                    print(f"[LLM:{eid}] responses ok, len={len(ans)}")
                    return re.sub(r"\s+", " ", ans)
                except Exception as e:
                    last_error = e
                    # 回退到 Chat
                    resp = call_chat_api()
                    ans = _extract_text_from_chat(resp)
                    if not ans:
                        raise RuntimeError("Empty response text from Chat API (fallback)")
                    print(f"[LLM:{eid}] chat fallback ok, len={len(ans)}")
                    return re.sub(r"\s+", " ", ans)

            # chat 优先（或强制 chat）
            try:
                resp = call_chat_api()
                ans = _extract_text_from_chat(resp)
                if not ans:
                    raise RuntimeError("Empty response text from Chat API")
                print(f"[LLM:{eid}] chat ok, len={len(ans)}")
                return re.sub(r"\s+", " ", ans)
            except Exception as e:
                last_error = e
                if llm_api == "chat":
                    # 仅 chat 模式则不回退
                    raise
                # auto 模式时回退 Responses
                resp = call_responses_api()
                ans = _extract_text_from_responses(resp)
                if not ans:
                    raise RuntimeError("Empty response text from Responses API (fallback)")
                print(f"[LLM:{eid}] responses fallback ok, len={len(ans)}")
                return re.sub(r"\s+", " ", ans)

        except Exception as e:
            last_error = e
            print(f"[LLM:{eid}] error (will retry): {e}")
            time.sleep(final_retry_delay)

    # 到此说明重试耗尽
    raise RuntimeError(f"Max retries exceeded. Last error: {last_error}")


# -------------------------
# 简单自测
# -------------------------
if __name__ == "__main__":
    cfg = load_llm_config()
    print("LLM config loaded with endpoints:", list(cfg.get("endpoints", {}).keys()))
    test_prompt = "List 3 typical symptoms of a cold."
    test_system = "You are a medical expert."
    # 需确保 test model 已在 model_map 映射
    response = query_model("gpt-5-mini", test_prompt, test_system)
    print("Response:", response)

    response2 = query_model("openbiollm70b", test_prompt, test_system)
    print("Response2:", response2)
