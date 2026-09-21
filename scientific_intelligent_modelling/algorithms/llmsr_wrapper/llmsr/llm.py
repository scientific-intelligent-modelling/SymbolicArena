import os
import time
"""统一的 LLM 客户端封装。

提供商/模型命名规则：'provider/model'，provider 大小写不敏感，model 保留大小写与路径。
当前支持：deepseek、siliconflow、deepinfra、ollama、blt、cstcloud（科技云）、mock。
"""
import requests
from typing import List, Dict, Tuple

# 单次实验级别的全局 token 统计（需由调用方在实验开始前手动 reset）
GLOBAL_TOKENS = {
    'prompt': 0,    # 提示词（prompt）部分 token
    'thinking': 0,  # 推理/思维链部分 token（reasoning_tokens）
    'content': 0,   # 可见输出部分 token（completion_tokens - reasoning_tokens）
    'total': 0,     # provider 返回的总 token（通常含 prompt + completion）
}
# 单次实验级别的全局时间统计（秒）
GLOBAL_TIME_SECONDS: float = 0.0


def reset_global_tokens():
    """重置本次实验的全局 token 统计。"""
    GLOBAL_TOKENS['prompt'] = 0
    GLOBAL_TOKENS['thinking'] = 0
    GLOBAL_TOKENS['content'] = 0
    GLOBAL_TOKENS['total'] = 0


def get_global_tokens() -> Dict[str, int]:
    """获取本次实验的全局 token 统计（thinking/content/total）。"""
    return dict(GLOBAL_TOKENS)


def reset_global_time():
    """重置本次实验的大模型总耗时统计（秒）。"""
    global GLOBAL_TIME_SECONDS
    GLOBAL_TIME_SECONDS = 0.0


def get_global_time() -> float:
    """获取本次实验的大模型总耗时（秒）。"""
    return float(GLOBAL_TIME_SECONDS)


class LLMClient:
    tokens = {
        'prompt': 0,
        'content': 0,
        'reasoning': 0,
        'total': 0,
    }

    def __init__(self, api_key: str, model: str, base_url: str):
        """
        初始化 LLM 客户端。

        :param api_key: API 密钥
        :param model: 模型名称
        :param base_url: API 的基础 URL
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        # 实例级别的累计统计（无需显式 reset；通常每个实验构造一个 client）
        self._call_index = 0
        self._cum_tokens = {
            'prompt': 0,
            'thinking': 0,
            'content': 0,
            'total': 0,
        }
        # 实例级别的累计耗时（秒）
        self._cum_time_seconds: float = 0.0
        self.kwargs = {
            'max_tokens': 1024,  # 更安全的默认值，避免超过部分模型上限
            'temperature': 0.6,
            'top_p': 0.3,
            'top_k': 30,
            'frequency_penalty': 0.5,
            'n': 1,
            'stream': False,
        }

    def _provider_name(self) -> str:
        try:
            url = (self.base_url or '').lower()
            if 'deepseek' in url:
                return 'deepseek'
            if 'siliconflow' in url or 'siliconflow.cn' in url:
                return 'siliconflow'
            if 'deepinfra' in url:
                return 'deepinfra'
            if 'bltcy' in url or 'blt' in url:
                return 'blt'
            if 'ollama' in url or 'localhost' in url:
                return 'ollama'
            if 'cstcloud' in url or 'uni-api.cstcloud.cn' in url:
                return 'cstcloud'
        except Exception:
            pass
        return 'llm'

    def chat(self, messages: List[Dict[str, str]]) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        request_url = f"{self.base_url.rstrip('/')}/chat/completions"

        model_name = self.model
        if 'qwen3' in model_name.lower():
            if '/think' in model_name:
                self.kwargs['enable_thinking'] = True
                model_name = model_name.replace('/think', '')
            else:
                self.kwargs['enable_thinking'] = False
                model_name = model_name.replace('/think', '')

        payload = {
            "model": model_name,
            "messages": messages,
        }
        # 仅透传 OpenAI Chat Completions 兼容字段，避免提供商拒绝未知参数
        allowed_keys = {
            'max_tokens', 'temperature', 'top_p', 'top_k', 'n', 'stream',
            'presence_penalty', 'frequency_penalty', 'stop', 'logprobs',
        }
        if isinstance(self.kwargs, dict):
            for k, v in self.kwargs.items():
                if k in allowed_keys:
                    payload[k] = v
        # 对输出 token 上限做保护（部分模型 4k 上限，统一取不超过 4096）
        try:
            if isinstance(payload.get('max_tokens'), int) and payload['max_tokens'] > 4096:
                payload['max_tokens'] = 4096
        except Exception:
            pass

        # 计时（用于统计每次调用与总耗时）
        start_time = time.time()
        try:
            response = requests.post(request_url, headers=headers, json=payload, timeout=120)
            # 状态码错误先抛出异常（下方 except 会打印详情）
            response.raise_for_status()
            # 尝试解析 JSON；失败时打印前 500 字符文本
            try:
                response_data = response.json()
            except ValueError:
                print("API 响应无法解析为 JSON，原始文本预览:", response.text[:500])
                raise

            # OpenAI 兼容接口错误格式：{"error": {...}}
            if isinstance(response_data, dict) and 'error' in response_data:
                err = response_data.get('error') or {}
                print("API 返回错误:", {
                    'type': err.get('type'),
                    'code': err.get('code'),
                    'message': err.get('message') or err,
                })
                raise requests.exceptions.HTTPError(f"API error: {err}")
            # end_time = time.time()

            # 保护性判断：缺少 choices 时打印提示
            if 'choices' not in response_data or not response_data['choices']:
                print("API 响应不包含 choices 字段或为空：", str(response_data)[:500])
                raise requests.exceptions.HTTPError("API response missing choices")

            message = response_data['choices'][0].get('message', {})
            content = message.get('content', '') or ''
            reasoning_content = message.get('reasoning_content', '') or ''

            # token统计
            usage = response_data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
            reasoning_tokens = 0
            if 'completion_tokens_details' in usage:
                reasoning_tokens = usage['completion_tokens_details'].get('reasoning_tokens', 0)

            self.tokens['prompt'] += prompt_tokens
            self.tokens['content'] += completion_tokens - reasoning_tokens
            self.tokens['reasoning'] += reasoning_tokens
            self.tokens['total'] += total_tokens

            # 更新单次实验全局统计
            try:
                GLOBAL_TOKENS['prompt'] += int(prompt_tokens)
                GLOBAL_TOKENS['thinking'] += int(reasoning_tokens)
                GLOBAL_TOKENS['content'] += int(completion_tokens - reasoning_tokens)
                GLOBAL_TOKENS['total'] += int(total_tokens)
            except Exception:
                pass

            # 实例级累计 & 全局时间累计 & 打印
            try:
                # 更新时间统计
                elapsed = time.time() - start_time
                self._cum_time_seconds += float(elapsed)
                try:
                    global GLOBAL_TIME_SECONDS
                    GLOBAL_TIME_SECONDS += float(elapsed)
                except Exception:
                    pass

                self._call_index += 1
                self._cum_tokens['prompt'] += int(prompt_tokens)
                self._cum_tokens['thinking'] += int(reasoning_tokens)
                self._cum_tokens['content'] += int(completion_tokens - reasoning_tokens)
                self._cum_tokens['total'] += int(total_tokens)

                provider = self._provider_name()
                header = f"[{provider}][{self.model}] 第{self._call_index}次"
                line_cur = (
                    f"本次 tokens：prompt={int(prompt_tokens)}, thinking={int(reasoning_tokens)}, "
                    f"content={int(completion_tokens - reasoning_tokens)}, total={int(total_tokens)}"
                )
                line_cum = (
                    f"累计 tokens：prompt={self._cum_tokens['prompt']}, thinking={self._cum_tokens['thinking']}, "
                    f"content={self._cum_tokens['content']}, total={self._cum_tokens['total']}"
                )
                line_time = (
                    f"本次用时：{elapsed:.2f}s，"
                    f"累计用时：{self._cum_time_seconds:.2f}s"
                )
                print(header + "\n" + line_cur + "\n" + line_cum + "\n" + line_time)
            except Exception:
                pass

            return {
                "content": content,
                "reasoning_content": reasoning_content,
                "tokens": {
                    "prompt": prompt_tokens,
                    "content": completion_tokens - reasoning_tokens,
                    "reasoning": reasoning_tokens,
                    "total": total_tokens
                }
            }

        except requests.exceptions.RequestException as e:
            print(f"通过 requests 调用 API 时出错: {e}")
            if e.response is not None:
                try:
                    print("错误详情(JSON):", e.response.json())
                except ValueError:
                    try:
                        print("错误详情(TEXT):", e.response.text[:500])
                    except Exception:
                        pass
            raise

class DeepSeekClient(LLMClient):
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.deepseek.com"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)

class SiliconflowClient(LLMClient):
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.siliconflow.cn/v1"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)

class DeepInfraClient(LLMClient):
    """DeepInfra，OpenAI Chat Completions 兼容接口。"""

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.deepinfra.com/v1/openai"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)

class CSTCloudClient(LLMClient):
    """CSTCloud（科技云）提供商，OpenAI Chat Completions 兼容接口。

    默认基址：https://uni-api.cstcloud.cn/v1
    使用示例：model="CSTCloud/gpt-oss-120b" 或 "CSTCloud/qwen3:235b"
    建议环境变量：CSTCLOUD_API_KEY
    """
    def __init__(self, api_key: str, model: str, base_url: str = "https://uni-api.cstcloud.cn/v1"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)

# 兼容旧拼写，避免历史引用报错
SliconflowClient = SiliconflowClient

class OllamaClient(LLMClient):
    def __init__(self, api_key: str, model: str, base_url: str = "http://localhost:11111/v1"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)

class BltClient(LLMClient):
    """BLT（柏拉图）网关，OpenAI Chat Completions 兼容接口。

    默认基址含 /v1，路径将拼接为 /chat/completions。
    """
    def __init__(self, api_key: str, model: str, base_url: str = None):
        base_url = base_url or os.getenv('BLT_API_BASE', 'https://api.bltcy.ai/v1')
        super().__init__(api_key=api_key, model=model, base_url=base_url)


class MockBaseClient(LLMClient):
    """mock 客户端基类：不发网络请求，只回放预设响应。"""

    def __init__(
        self,
        model: str,
        sleep_seconds: float = 0.0,
        prompt_tokens: int = 0,
        content_tokens: int = 0,
    ):
        super().__init__(api_key="", model=model, base_url="mock://local")
        self.sleep_seconds = max(float(sleep_seconds), 0.0)
        self.prompt_tokens = max(int(prompt_tokens), 0)
        self.content_tokens = max(int(content_tokens), 0)

    def _record_usage(self) -> None:
        total_tokens = self.prompt_tokens + self.content_tokens
        self.tokens['prompt'] += self.prompt_tokens
        self.tokens['content'] += self.content_tokens
        self.tokens['reasoning'] += 0
        self.tokens['total'] += total_tokens

        try:
            GLOBAL_TOKENS['prompt'] += int(self.prompt_tokens)
            GLOBAL_TOKENS['thinking'] += 0
            GLOBAL_TOKENS['content'] += int(self.content_tokens)
            GLOBAL_TOKENS['total'] += int(total_tokens)
        except Exception:
            pass

        try:
            self._call_index += 1
            self._cum_tokens['prompt'] += int(self.prompt_tokens)
            self._cum_tokens['thinking'] += 0
            self._cum_tokens['content'] += int(self.content_tokens)
            self._cum_tokens['total'] += int(total_tokens)
            provider = 'mock'
            header = f"[{provider}][{self.model}] 第{self._call_index}次"
            line_cur = (
                f"本次 tokens：prompt={int(self.prompt_tokens)}, thinking=0, "
                f"content={int(self.content_tokens)}, total={int(total_tokens)}"
            )
            line_cum = (
                f"累计 tokens：prompt={self._cum_tokens['prompt']}, thinking={self._cum_tokens['thinking']}, "
                f"content={self._cum_tokens['content']}, total={self._cum_tokens['total']}"
            )
            line_time = (
                f"本次用时：{self.sleep_seconds:.2f}s，"
                f"累计用时：{self._cum_time_seconds + self.sleep_seconds:.2f}s"
            )
            self._cum_time_seconds += float(self.sleep_seconds)
            try:
                global GLOBAL_TIME_SECONDS
                GLOBAL_TIME_SECONDS += float(self.sleep_seconds)
            except Exception:
                pass
            print(header + "\n" + line_cur + "\n" + line_cum + "\n" + line_time)
        except Exception:
            pass

    def _build_response(self, content: str) -> dict:
        total_tokens = self.prompt_tokens + self.content_tokens
        self._record_usage()
        return {
            "content": content,
            "reasoning_content": "",
            "tokens": {
                "prompt": self.prompt_tokens,
                "content": self.content_tokens,
                "reasoning": 0,
                "total": total_tokens,
            }
        }


class MockFixedClient(MockBaseClient):
    """固定返回同一段响应，用于并发压测，完全不发起真实网络请求。"""

    def __init__(
        self,
        model: str,
        fixed_response: str,
        sleep_seconds: float = 0.0,
        prompt_tokens: int = 0,
        content_tokens: int = 0,
    ):
        super().__init__(
            model=model,
            sleep_seconds=sleep_seconds,
            prompt_tokens=prompt_tokens,
            content_tokens=content_tokens,
        )
        self.fixed_response = fixed_response

    def chat(self, messages: List[Dict[str, str]]) -> dict:
        del messages
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        return self._build_response(self.fixed_response)


class MockPoolClient(MockBaseClient):
    """轮询返回一个固定响应池，用于多轮压力测试。"""

    def __init__(
        self,
        model: str,
        responses: List[str],
        sleep_seconds: float = 0.0,
        prompt_tokens: int = 0,
        content_tokens: int = 0,
    ):
        super().__init__(
            model=model,
            sleep_seconds=sleep_seconds,
            prompt_tokens=prompt_tokens,
            content_tokens=content_tokens,
        )
        cleaned = [str(item) for item in responses if str(item).strip()]
        if not cleaned:
            raise ValueError("mock/pool 需要至少一条非空响应")
        self.responses = cleaned
        self._response_index = 0

    def chat(self, messages: List[Dict[str, str]]) -> dict:
        del messages
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        content = self.responses[self._response_index % len(self.responses)]
        self._response_index += 1
        return self._build_response(content)

def parse_provider_model(model_str: str) -> Tuple[str, str]:
    """
    解析模型字符串为 (provider, model)。

    规则：第一个 '/' 之前为提供商（大小写不敏感），之后的全部为模型名（大小写敏感，允许包含 '/').
    示例：
    - "deepseek/deepseek-chat" -> ("deepseek", "deepseek-chat")
    - "SiliconFlow/Qwen/Qwen3-8B" -> ("siliconflow", "Qwen/Qwen3-8B")
    - "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct" -> ("deepinfra", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    - "ollama/llama3.1:8b" -> ("ollama", "llama3.1:8b")
    - "mock/fixed" -> ("mock", "fixed")
    - "mock/pool" -> ("mock", "pool")
    """
    if not isinstance(model_str, str) or '/' not in model_str:
        raise ValueError("缺少模型提供商：请使用 'provider/model' 格式，例如 'CSTCloud/gpt-oss-120b'")
    provider, model = model_str.split('/', 1)
    return provider.lower(), model

class ClientFactory:
    @staticmethod
    def from_config(config: dict):
        """
        基于 'provider/model' 创建具体客户端。

        必填：config['model']（形如 'provider/model'）。
        选填：config['api_key']、config['base_url']。
        """
        if 'model' not in config:
            raise ValueError("缺少必要字段: model")

        provider, model = parse_provider_model(config['model'])
        api_key_cfg = config.get('api_key')
        base_url = config.get('base_url')

        # api_key 支持：
        # 1) 字符串（兼容旧格式）
        # 2) 字典：可按 provider 或完整 model（'provider/model'）配置不同 key
        api_key = None
        if isinstance(api_key_cfg, dict):
            def _get_case_insensitive(d: dict, k: str):
                for kk, vv in d.items():
                    try:
                        if str(kk).lower() == str(k).lower():
                            return vv
                    except Exception:
                        pass
                return None
            # 优先匹配完整模型名，其次按提供商名
            api_key = _get_case_insensitive(api_key_cfg, config.get('model', '')) or _get_case_insensitive(api_key_cfg, provider)
        elif isinstance(api_key_cfg, str):
            api_key = api_key_cfg
        else:
            api_key = None

        def _require_api_key(value: str, env_name: str) -> str:
            resolved = value or os.getenv(env_name, '')
            if not isinstance(resolved, str) or not resolved.strip():
                raise ValueError(
                    f"LLM provider '{provider}' 缺少 API key："
                    f"请在 llm.config 的 api_key 字段或环境变量 {env_name} 中配置。"
                )
            return resolved

        # 设置默认 base_url
        if provider == 'deepseek':
            base_url = base_url or "https://api.deepseek.com"
            return DeepSeekClient(api_key=_require_api_key(api_key, 'DEEPSEEK_API_KEY'), model=model, base_url=base_url)
        elif provider in ('siliconflow', 'silicon-flow', 'sflow'):
            base_url = base_url or "https://api.siliconflow.cn/v1"
            return SiliconflowClient(api_key=_require_api_key(api_key, 'SILICONFLOW_API_KEY'), model=model, base_url=base_url)
        elif provider in ('deepinfra', 'deep-infra'):
            base_url = base_url or "https://api.deepinfra.com/v1/openai"
            return DeepInfraClient(api_key=_require_api_key(api_key, 'DEEPINFRA_API_KEY'), model=model, base_url=base_url)
        elif provider == 'ollama':
            base_url = base_url or "http://localhost:11111/v1"
            return OllamaClient(api_key=api_key or '', model=model, base_url=base_url)
        elif provider in ('blt', 'bltcy', 'plato'):
            # 优先使用传入 api_key，否则读环境变量 BLT_API_KEY
            return BltClient(api_key=_require_api_key(api_key, 'BLT_API_KEY'), model=model, base_url=base_url or os.getenv('BLT_API_BASE', 'https://api.bltcy.ai/v1'))
        elif provider in ('cstcloud', 'cst', 'cst-cloud', 'keji', 'keji-yun'):
            # 科技云：默认基址 https://uni-api.cstcloud.cn/v1
            return CSTCloudClient(api_key=_require_api_key(api_key, 'CSTCLOUD_API_KEY'), model=model, base_url=base_url or 'https://uni-api.cstcloud.cn/v1')
        elif provider in ('mock', 'fake'):
            if model == 'pool':
                responses = (
                    config.get('mock_responses')
                    or config.get('response_pool')
                    or config.get('responses')
                    or [
                        "    return x0 + params[0]\n",
                        "    return x1 + params[0]\n",
                        "    return x0 + x1 + params[0]\n",
                    ]
                )
                return MockPoolClient(
                    model=model,
                    responses=responses,
                    sleep_seconds=config.get('mock_sleep_seconds', 0.0),
                    prompt_tokens=config.get('mock_prompt_tokens', 0),
                    content_tokens=config.get('mock_content_tokens', 0),
                )
            fixed_response = (
                config.get('fixed_response')
                or config.get('mock_response')
                or "    return x0 + params[0]\n"
            )
            return MockFixedClient(
                model=model,
                fixed_response=fixed_response,
                sleep_seconds=config.get('mock_sleep_seconds', 0.0),
                prompt_tokens=config.get('mock_prompt_tokens', 0),
                content_tokens=config.get('mock_content_tokens', 0),
            )
        else:
            raise ValueError(f"不支持的提供商: {provider}，请使用 'deepseek'、'siliconflow'、'deepinfra'、'blt'、'cstcloud'、'mock' 或 'ollama'")
        


if __name__ == '__main__':
    # 确保你的 API 密钥已经设置为环境变量 SILICONFLOW_API_KEY
    # 或者直接在这里替换 "your-siliconflow-api-key"


    client = OllamaClient(api_key='', model='llama3.1:8b')
    messages = [
        {"role": "user", "content": "你好，请介绍一下你自己，并说明你的思考过程。"}
    ]
    response_content = client.chat(messages)
    print(response_content)

    # deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key")
    # if deepseek_api_key == "your-deepseek-api-key":
    #     print("请设置 DEEPSEEK_API_KEY 环境变量或直接在代码中提供您的 API 密钥。")
    # else:
    #     client = DeepSeekClient(api_key=deepseek_api_key, model='deepseek-reasoner')
    #     messages = [
    #         {"role": "user", "content": "你好，请介绍一下你自己，并说明你的思考过程。"}
    #     ]
    #     response_content = client.chat(messages)
    #     print(response_content)

    # print('=='*20)

    # api_key = os.getenv("SILICONFLOW_API_KEY", "your-siliconflow-api-key")
    # if api_key == "your-siliconflow-api-key":
    #     print("请设置 SILICONFLOW_API_KEY 环境变量或直接在代码中提供您的 API 密钥。")
    # else:
    #     model_lists = [
    #         'Qwen/Qwen3-8B/think',
    #         'Qwen/Qwen3-8B',
    #         'Qwen/QwQ-32B',
    #         'Qwen/Qwen3-32B',
    #         'Qwen/Qwen2.5-72B-Instruct',
    #         'Qwen/Qwen2.5-32B-Instruct',
    #     ]
    #     for model in model_lists:
    #         print('【this is model: 】', model)
    #         client = SliconflowClient(api_key=api_key, model=model)
    #         messages = [
    #             {"role": "user", "content": "你好，请介绍一下你自己，并说明你的思考过程。"}
    #         ]
            
    #         try:
    #             response_content = client.chat(messages)
    #             print(response_content)
    #         except Exception as e:
    #             print(f"调用模型时出错: {e}")

    #         print("\n" + "="*20 + "\n")
