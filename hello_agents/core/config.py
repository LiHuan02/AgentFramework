"""配置管理"""
import os
from typing import Optional
from pydantic import BaseModel

class Config(BaseModel):
    """HelloAgents配置类"""
    
    # LLM 生成参数
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
    # 系统配置
    debug: bool = False
    log_level: str = "INFO"
    max_history_length: int = 100
    
    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置"""
        max_tokens_str = os.getenv("MAX_TOKENS")
        return cls(
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(max_tokens_str) if max_tokens_str else None,
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return self.model_dump()
