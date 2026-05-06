import os
import time
"""note LLM note 

note/note 'provider/model' provider note model note 
current note deepseek siliconflow deepinfra openai ollama blt cstcloud 
"""
import requests
from typing import List, Dict, Tuple

# note token note note reset 
GLOBAL_TOKENS = {
    'prompt': 0,    # note prompt note token
    'thinking': 0,  # note/note token reasoning_tokens 
    'content': 0,   # note token completion_tokens - reasoning_tokens 
    'total': 0,     # provider note token note prompt + completion 
}
GLOBAL_TIME_SECONDS: float = 0.0

def reset_global_tokens():
    """note token note """
    GLOBAL_TOKENS['prompt'] = 0
    GLOBAL_TOKENS['thinking'] = 0
    GLOBAL_TOKENS['content'] = 0
    GLOBAL_TOKENS['total'] = 0

def get_global_tokens() -> Dict[str, int]:
    """note token note thinking/content/total  """
    return dict(GLOBAL_TOKENS)


def reset_global_time():
    """note note  """
    global GLOBAL_TIME_SECONDS
    GLOBAL_TIME_SECONDS = 0.0


def get_global_time() -> float:
    """note note  """
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
        note LLM note 

        :param api_key: API note
        :param model: note
        :param base_url: API note URL
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        # note note reset note client 
        self._call_index = 0
        self._cum_tokens = {
            'prompt': 0,
            'thinking': 0,
            'content': 0,
            'total': 0,
        }
        self._cum_time_seconds: float = 0.0
        self.kwargs = {
            'max_tokens': 1024,  # note default note note
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
        # note OpenAI Chat Completions notefield note
        allowed_keys = {
            'max_tokens', 'temperature', 'top_p', 'top_k', 'n', 'stream',
            'presence_penalty', 'frequency_penalty', 'stop', 'logprobs',
        }
        if isinstance(self.kwargs, dict):
            for k, v in self.kwargs.items():
                if k in allowed_keys:
                    payload[k] = v
        # note token note note 4k note note 4096 
        try:
            if isinstance(payload.get('max_tokens'), int) and payload['max_tokens'] > 4096:
                payload['max_tokens'] = 4096
        except Exception:
            pass

        start_time = time.time()
        try:
            response = requests.post(request_url, headers=headers, json=payload, timeout=120)
            # note note except note 
            response.raise_for_status()
            # note JSON note 500 note
            try:
                response_data = response.json()
            except ValueError:
                print("API response note JSON raw text preview:", response.text[:500])
                raise

            # OpenAI note {"error": {...}}
            if isinstance(response_data, dict) and 'error' in response_data:
                err = response_data.get('error') or {}
                print("API returned error:", {
                    'type': err.get('type'),
                    'code': err.get('code'),
                    'message': err.get('message') or err,
                })
                raise requests.exceptions.HTTPError(f"API error: {err}")
            # end_time = time.time()

            # note note choices note
            if 'choices' not in response_data or not response_data['choices']:
                print("API response note choices fieldnote ", str(response_data)[:500])
                raise requests.exceptions.HTTPError("API response missing choices")

            message = response_data['choices'][0].get('message', {})
            content = message.get('content', '') or ''
            reasoning_content = message.get('reasoning_content', '') or ''

            # tokennote
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

            # note
            try:
                GLOBAL_TOKENS['prompt'] += int(prompt_tokens)
                GLOBAL_TOKENS['thinking'] += int(reasoning_tokens)
                GLOBAL_TOKENS['content'] += int(completion_tokens - reasoning_tokens)
                GLOBAL_TOKENS['total'] += int(total_tokens)
            except Exception:
                pass

            # note note
            try:
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
                header = f"[{provider}][{self.model}] call {self._call_index}"
                line_cur = (
                    f"current tokens prompt={int(prompt_tokens)}, thinking={int(reasoning_tokens)}, "
                    f"content={int(completion_tokens - reasoning_tokens)}, total={int(total_tokens)}"
                )
                line_cum = (
                    f"cumulative tokens prompt={self._cum_tokens['prompt']}, thinking={self._cum_tokens['thinking']}, "
                    f"content={self._cum_tokens['content']}, total={self._cum_tokens['total']}"
                )
                line_time = (
                    f"current elapsed time {elapsed:.2f}s "
                    f"cumulative elapsed time {self._cum_time_seconds:.2f}s"
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
            print(f"error while calling the API through requests: {e}")
            if e.response is not None:
                try:
                    print("error details(JSON):", e.response.json())
                except ValueError:
                    try:
                        print("error details(TEXT):", e.response.text[:500])
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
    """DeepInfra OpenAI Chat Completions note """

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.deepinfra.com/v1/openai"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)

# note note
SliconflowClient = SiliconflowClient

class OllamaClient(LLMClient):
    def __init__(self, api_key: str, model: str, base_url: str = "http://localhost:11111/v1"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)

class BltClient(LLMClient):
    """BLT note note OpenAI Chat Completions note 

    default note /v1 note /chat/completions 
    """
    def __init__(self, api_key: str, model: str, base_url: str = None):
        base_url = base_url or os.getenv('BLT_API_BASE', 'https://api.bltcy.ai/v1')
        super().__init__(api_key=api_key, model=model, base_url=base_url)


class CSTCloudClient(LLMClient):
    """CSTCloud note OpenAI Chat Completions note """

    def __init__(self, api_key: str, model: str, base_url: str = "https://uni-api.cstcloud.cn/v1"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)


class OpenAIClient(LLMClient):
    """OpenAI Chat Completions note """

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)


def parse_provider_model(model_str: str) -> Tuple[str, str]:
    """
    note (provider, model) 

    note note '/' note note  note note note '/').
    note 
    - "deepseek/deepseek-chat" -> ("deepseek", "deepseek-chat")
    - "SiliconFlow/Qwen/Qwen3-8B" -> ("siliconflow", "Qwen/Qwen3-8B")
    - "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct" -> ("deepinfra", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    - "ollama/llama3.1:8b" -> ("ollama", "llama3.1:8b")
    """
    if not isinstance(model_str, str) or '/' not in model_str:
        raise ValueError("invalid model-name format note 'provider/model' note 'deepseek/deepseek-chat'")
    provider, model = model_str.split('/', 1)
    return provider.lower(), model

class ClientFactory:
    @staticmethod
    def from_config(config: dict):
        """
        note 'provider/model' note 

        note config['model'] note 'provider/model'  
        note config['api_key'] config['base_url'] 
        """
        if 'model' not in config:
            raise ValueError("missing required field: model")

        provider, model = parse_provider_model(config['model'])
        api_key = config.get('api_key')
        base_url = config.get('base_url')

        # note default base_url
        if provider == 'deepseek':
            base_url = base_url or "https://api.deepseek.com"
            return DeepSeekClient(api_key=api_key or os.getenv('DEEPSEEK_API_KEY', ''), model=model, base_url=base_url)
        elif provider in ('siliconflow', 'silicon-flow', 'sflow'):
            base_url = base_url or "https://api.siliconflow.cn/v1"
            return SiliconflowClient(api_key=api_key or os.getenv('SILICONFLOW_API_KEY', ''), model=model, base_url=base_url)
        elif provider in ('deepinfra', 'deep-infra'):
            base_url = base_url or "https://api.deepinfra.com/v1/openai"
            return DeepInfraClient(api_key=api_key or os.getenv('DEEPINFRA_API_KEY', ''), model=model, base_url=base_url)
        elif provider == 'openai':
            base_url = base_url or "https://api.openai.com/v1"
            return OpenAIClient(api_key=api_key or os.getenv('OPENAI_API_KEY', ''), model=model, base_url=base_url)
        elif provider == 'ollama':
            base_url = base_url or "http://localhost:11111/v1"
            return OllamaClient(api_key=api_key or '', model=model, base_url=base_url)
        elif provider in ('blt', 'bltcy', 'plato'):
            # prefernote api_key noteenvironment note BLT_API_KEY
            return BltClient(api_key=api_key or os.getenv('BLT_API_KEY', ''), model=model, base_url=base_url or os.getenv('BLT_API_BASE', 'https://api.bltcy.ai/v1'))
        elif provider in ('cstcloud', 'cst', 'cst-cloud', 'keji', 'keji-yun'):
            base_url = base_url or "https://uni-api.cstcloud.cn/v1"
            return CSTCloudClient(api_key=api_key or os.getenv('CSTCLOUD_API_KEY', ''), model=model, base_url=base_url)
        else:
            raise ValueError(f"unsupported provider: {provider} note 'deepseek' 'siliconflow' 'deepinfra' 'openai' 'blt' 'cstcloud' note 'ollama'")
        


if __name__ == '__main__':
    # note API noteenvironment note SILICONFLOW_API_KEY
    # note "your-siliconflow-api-key"


    client = OllamaClient(api_key='', model='llama3.1:8b')
    messages = [
        {"role": "user", "content": "note note note "}
    ]
    response_content = client.chat(messages)
    print(response_content)

    # deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key")
    # if deepseek_api_key == "your-deepseek-api-key":
    #     print("please set DEEPSEEK_API_KEY environment note API note ")
    # else:
    #     client = DeepSeekClient(api_key=deepseek_api_key, model='deepseek-reasoner')
    #     messages = [
    #         {"role": "user", "content": "note note note "}
    #     ]
    #     response_content = client.chat(messages)
    #     print(response_content)

    # print('=='*20)

    # api_key = os.getenv("SILICONFLOW_API_KEY", "your-siliconflow-api-key")
    # if api_key == "your-siliconflow-api-key":
    #     print("please set SILICONFLOW_API_KEY environment note API note ")
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
    #         print(' this is model:  ', model)
    #         client = SliconflowClient(api_key=api_key, model=model)
    #         messages = [
    #             {"role": "user", "content": "note note note "}
    #         ]
            
    #         try:
    #             response_content = client.chat(messages)
    #             print(response_content)
    #         except Exception as e:
    #             print(f"error while calling model: {e}")

    #         print("\n" + "="*20 + "\n")
