import os
from typing import Dict, Optional
from langchain_openai import ChatOpenAI

try:
    # When imported as `core.llm`
    from .config import Config  # type: ignore
except ImportError:
    # When executed directly: `python core/llm.py`
    from config import Config
# from dotenv import load_dotenv



# load_dotenv()

# api_key = os.getenv("DEEPSEEK_API_KEY")
# base_url = os.getenv("DEEPSEEK_BASE_URL")

# 允许覆盖的参数列表，避免传入无效字段
_ALLOWED_PARAMS = {
    "model_name",
    "temperature",
    "max_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
}

class Model:
    """
    一个用于与 LLM 交互的模型类。
    参数:
        api_key (str): 用于认证的 API 密钥。
        base_url (str): LLM 服务的基础 URL。
    属性:
        model_name (str): 使用的模型名称。
        temperature (float): 控制生成文本的随机性。
        max_tokens (int): 生成文本的最大长度。
        top_p (float): 用于 nucleus 采样的概率阈值。
        frequency_penalty (float): 控制重复词汇的惩罚程度。
        presence_penalty (float): 控制新话题引入的惩罚程度。
    方法:
        create(): 初始化并返回一个 LLM 客户端实例。
    """
    def __init__(self, api_key: str, base_url: str):
        if not api_key:
            raise ValueError("❌ 未找到 DEEPSEEK_API_KEY，请检查 .env 文件！")
        if not base_url:
            raise ValueError("❌ 未找到 DEEPSEEK_BASE_URL，请检查 .env 文件！")
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = "deepseek-chat"
        self.temperature = 1
        self.max_tokens = 2048
        self.top_p = 1.0
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
    
    def create(self):
        return ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
        )
    
    def set_model_name(self, model_name: str):
        self.model_name = model_name
    
    def set_temperature(self, temperature: float):
        self.temperature = temperature
        
    def set_max_tokens(self, max_tokens: int):
        self.max_tokens = max_tokens
        
    def set_top_p(self, top_p: float):
        self.top_p = top_p
        
    def set_frequency_penalty(self, frequency_penalty: float):
        self.frequency_penalty = frequency_penalty  
        
    def set_presence_penalty(self, presence_penalty: float):
        self.presence_penalty = presence_penalty    
        
    def __repr__(self):
        return (f"Model(model_name={self.model_name}, temperature={self.temperature}, "
                f"max_tokens={self.max_tokens}, top_p={self.top_p}, "
                f"frequency_penalty={self.frequency_penalty}, presence_penalty={self.presence_penalty})")
    

# if __name__ == "__main__":
#     config = Config()
#     model = Model(api_key=config.get_api_key(), base_url=config.get_base_url())
#     model.set_temperature(2)
#     llm_client = model.create()
#     print("LLM 客户端已创建:", llm_client)
