# LLM.py
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional

# 加载 .env 文件中的环境变量
load_dotenv()

class LLM:
    """
    自定义LLM客户端，支持不同提供商（Openai、ModelScope）
    """

    # 简单的provider映射
    provider_defaults = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "env_key": "OPENAI_API_KEY",
        },
        "modelscope": {
            "base_url": "https://api-inference.modelscope.cn/v1/",
            "env_key": "MODELSCOPE_API_KEY",
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "env_key": None,  # 不需要API key
            "default_model": "llama3"
        },
        "vllm": {
            "base_url": "http://localhost:8000/v1",
            "env_key": None,
            "default_model": "Qwen/Qwen1.5-0.5B-Chat"
        }
    }

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        
        # 如果指定了provider，使用对应的默认配置
        if provider and provider in self.provider_defaults:
            defaults = self.provider_defaults[provider]
            self.base_url = base_url or os.getenv("LLM_BASE_URL") or defaults["base_url"]
            self.api_key = api_key or os.getenv(defaults["env_key"]) or os.getenv("LLM_API_KEY")
        else:
            # 否则使用通用配置
            self.base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
            self.api_key = api_key or os.getenv("LLM_API_KEY")
        
        self.model = model or os.getenv("LLM_MODEL_ID")
        
        if not self.api_key:
            raise ValueError("API密钥未提供")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None


