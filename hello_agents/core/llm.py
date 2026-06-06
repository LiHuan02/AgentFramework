"""HelloAgents统一LLM接口 - 基于OpenAI原生API"""

import os
from typing import Literal, Optional, Iterator
from openai import OpenAI

from .exceptions import HelloAgentsException

# 支持的LLM提供商
SUPPORTED_PROVIDERS = Literal[
    "openai",
    "deepseek",
    "qwen",
    "modelscope",
    "kimi",
    "zhipu",
    "ollama",
    "vllm",
    "local",
    "custom",
]

# 供应商默认配置映射表
PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini"
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat"
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus"
    },
    "modelscope": {
        "base_url": "https://api-inference.modelscope.cn/v1/",
        "model": "Qwen/Qwen2.5-72B-Instruct"
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k"
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash"
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  # 本地服务无需真实Key的占位符
        "model": "llama3.2"
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "api_key": "vllm",
        "model": "meta-llama/Llama-2-7b-chat-hf"
    },
    "local": {
        "base_url": "http://localhost:8000/v1",
        "api_key": "local",
        "model": "local-model"
    },
}

class HelloAgentsLLM:
    """
    为HelloAgents定制的LLM客户端。
    仅依赖 LLM_MODEL_ID, LLM_API_KEY, LLM_BASE_URL, LLM_TIMEOUT 四个环境变量。
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[SUPPORTED_PROVIDERS] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ):
        # 基础参数优先级：显式传入 > 环境变量
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        self.kwargs = kwargs

        # 1. 确定 provider：显式指定 > base_url智能推断
        self.provider = provider or self._auto_detect_provider(base_url)

        # 2. 解析凭证：显式传入 > 环境变量 > 供应商默认值
        self.api_key, self.base_url = self._resolve_credentials(api_key, base_url)

        # 3. 模型兜底：如果此时仍缺模型，使用供应商默认模型
        if not self.model:
            self.model = PROVIDER_DEFAULTS.get(self.provider, {}).get("model", "gpt-4o-mini")

        # 4. 校验核心参数
        if not all([self.api_key, self.base_url]):
            raise HelloAgentsException(
                "API密钥和服务地址必须被提供或在.env文件中定义（LLM_API_KEY / LLM_BASE_URL）。"
            )

        # 5. 创建客户端
        self._client = self._create_client()

    def _auto_detect_provider(self, base_url: Optional[str]) -> str:
        """仅依赖 base_url 智能推断供应商"""
        url = (base_url or os.getenv("LLM_BASE_URL", "")).lower()
        
        if not url or "api.openai.com" in url:
            return "openai"
        if "api.deepseek.com" in url:
            return "deepseek"
        if "dashscope.aliyuncs.com" in url:
            return "qwen"
        if "api-inference.modelscope.cn" in url:
            return "modelscope"
        if "api.moonshot.cn" in url:
            return "kimi"
        if "open.bigmodel.cn" in url:
            return "zhipu"
        if ":11434" in url or "ollama" in url:
            return "ollama"
        if ":8000" in url or "vllm" in url:
            return "vllm"
        if "localhost" in url or "127.0.0.1" in url:
            return "local"
            
        return "custom"

    def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple[str, str]:
        """统一解析 API Key 和 Base URL，消除了冗长的 if-else"""
        defaults = PROVIDER_DEFAULTS.get(self.provider, {})
        
        # 优先级：显式传入参数 > .env环境变量 > 供应商默认配置
        resolved_api_key = api_key or os.getenv("LLM_API_KEY") or defaults.get("api_key")
        resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or defaults.get("base_url")
        
        return resolved_api_key, resolved_base_url

    def _create_client(self) -> OpenAI:
        """创建OpenAI客户端"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        """流式调用LLM，逐步返回文本片段"""
        print(f"🧠 正在调用 {self.model} 模型 (流式)...")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stream=True,
            )

            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
            print()  # 结束换行
            
        except Exception as e:
            print(f"\n❌ 调用LLM API时发生错误: {e}")
            raise HelloAgentsException(f"LLM调用失败: {str(e)}")

    def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
        """非流式调用LLM，返回完整响应字符串"""
        print(f"🧠 正在调用 {self.model} 模型 (非流式)...")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise HelloAgentsException(f"LLM调用失败: {str(e)}")
