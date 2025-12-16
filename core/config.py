import os
from dotenv import load_dotenv

class Config:
    """
    配置类，用于加载和存储应用程序的配置参数。
    属性:
        api_key (str): 用于认证的 API 密钥。
        base_url (str): LLM 服务的基础 URL。   
    方法:
        load(): 从环境变量或配置文件中加载配置参数。
    """
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL")
        self.chat_model = os.getenv("DEEPSEEK_MODEL") or os.getenv("LLM_MODEL") or "deepseek-chat"
        self.embedding_model = os.getenv("EMBEDDING_MODEL")
        if not self.api_key:
            raise ValueError("❌ 未找到 DEEPSEEK_API_KEY，请检查 .env 文件！")
        if not self.base_url:
            raise ValueError("❌ 未找到 DEEPSEEK_BASE_URL，请检查 .env 文件！")
        if not self.embedding_model:
            raise ValueError("❌ 未找到 EMBEDDING_MODEL，请检查 .env 文件！")

    def get_api_key(self) -> str:
        return self.api_key
    
    def get_base_url(self) -> str:
        return self.base_url
    
    def get_embedding_model(self) -> str:
        return self.embedding_model

    def get_chat_model(self) -> str:
        return self.chat_model
        

if __name__ == "__main__":
    config = Config()
    print("API Key:", config.get_api_key())
    print("Base URL:", config.get_base_url())
    print("embedding_model:",config.get_embedding_model())
