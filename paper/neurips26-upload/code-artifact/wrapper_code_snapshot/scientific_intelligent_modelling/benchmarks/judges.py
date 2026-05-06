"""Benchmark 级别的符号一致性 Judge。

当前主要面向 LLM-SRBench 的 symbolic accuracy 场景：
- 输入 gold equation 与 predicted equation/program。
- 调用 OpenAI-compatible LLM 做等价性判断。
- 带 JSON 缓存，避免重复消耗 API。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from scientific_intelligent_modelling.srkit.llm import ClientFactory


PROMPT_VERSION = "llm_srbench_symbolic_judge_v1"


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"```(?:json|python|text)?", "", text)
    text = text.replace("```", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _answer_to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"yes", "true", "equivalent", "1"}:
            return True
        if v in {"no", "false", "not_equivalent", "0"}:
            return False
    return None


class LLMSymbolicJudge:
    """带缓存的 LLM 符号一致性评估器。"""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        cache_path: str | Path | None = None,
    ):
        self.config = dict(config)
        self.cache_path = Path(cache_path or self.config.get("cache_path") or "artifacts/benchmark_symbolic_judge_cache.json")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_cache()
        self._client = None

    @classmethod
    def from_config_path(cls, config_path: str | Path) -> "LLMSymbolicJudge":
        config_path = Path(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(config=config, cache_path=config.get("cache_path"))

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_cache(self) -> None:
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get_client(self):
        if self._client is None:
            self._client = ClientFactory.from_config(self.config)
        return self._client

    def available(self) -> bool:
        model = self.config.get("model")
        if not model:
            return False
        api_key = self.config.get("api_key")
        if isinstance(api_key, str) and api_key.strip():
            return True
        provider = str(model).split("/", 1)[0].lower()
        env_map = {
            "openai": "OPENAI_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "siliconflow": "SILICONFLOW_API_KEY",
            "blt": "BLT_API_KEY",
            "bltcy": "BLT_API_KEY",
            "plato": "BLT_API_KEY",
            "ollama": None,
        }
        env_name = env_map.get(provider)
        if env_name is None:
            return provider == "ollama"
        import os

        return bool(os.getenv(env_name, "").strip())

    def _build_prompt(
        self,
        *,
        gold_equation: str,
        predicted_equation: str,
    ) -> str:
        return f"""
You are evaluating symbolic equivalence for scientific equation discovery.

Task:
Decide whether the hypothesis can be mathematically equivalent to the ground-truth expression,
allowing free scalar constants and coefficient parameters in the hypothesis when appropriate.

Rules:
1. Focus on mathematical equivalence, not superficial syntax.
2. Ignore formatting differences, comments, code fences, and variable naming style.
3. If the hypothesis is a program, focus on the equation logic only.
4. Answer conservatively. If equivalence is unclear, answer false.
5. Return strict JSON only.

Ground truth:
{gold_equation}

Hypothesis:
{predicted_equation}

Return JSON with this schema:
{{
  "equivalent": true or false,
  "reasoning": "brief explanation",
  "confidence": 0.0
}}
""".strip()

    def _cache_key(self, *, gold_equation: str, predicted_equation: str) -> str:
        raw = json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "model": self.config.get("model"),
                "gold_equation": _normalize_text(gold_equation),
                "predicted_equation": _normalize_text(predicted_equation),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def judge(
        self,
        *,
        gold_equation: str,
        predicted_equation: str,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        gold_equation = _normalize_text(gold_equation)
        predicted_equation = _normalize_text(predicted_equation)
        key = self._cache_key(gold_equation=gold_equation, predicted_equation=predicted_equation)

        if use_cache and key in self._cache:
            cached = dict(self._cache[key])
            cached["cached"] = True
            return cached

        if not self.available():
            raise RuntimeError("LLM judge 当前不可用：缺少 model 或 API key")

        client = self._get_client()
        prompt = self._build_prompt(
            gold_equation=gold_equation,
            predicted_equation=predicted_equation,
        )
        response = client.chat([{"role": "user", "content": prompt}])
        content = response.get("content", "") if isinstance(response, dict) else str(response)
        parsed = _extract_json_object(content)

        result = {
            "equivalent": None,
            "reasoning": None,
            "confidence": None,
            "raw_response": content,
            "cached": False,
            "prompt_version": PROMPT_VERSION,
        }

        if parsed is not None:
            result["equivalent"] = _answer_to_bool(parsed.get("equivalent") or parsed.get("answer"))
            result["reasoning"] = parsed.get("reasoning")
            result["confidence"] = parsed.get("confidence")
        else:
            bool_guess = _answer_to_bool(content)
            result["equivalent"] = bool_guess
            result["reasoning"] = content[:500]

        self._cache[key] = {k: v for k, v in result.items() if k != "cached"}
        self._save_cache()
        return result


def llm_srbench_symbolic_accuracy(
    gold_equation: str,
    predicted_equation: str,
    *,
    judge: LLMSymbolicJudge,
) -> dict[str, Any]:
    """对单个样本计算 LLM-SRBench 风格 symbolic accuracy。"""
    judged = judge.judge(
        gold_equation=gold_equation,
        predicted_equation=predicted_equation,
    )
    return {
        "symbolic_accuracy": 1.0 if judged.get("equivalent") else 0.0,
        "judge_result": judged,
    }
