from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

# 消息角色字面量
MessageRole = Literal["user", "assistant", "system", "tool"]

class Message(BaseModel):
    """消息类，与 OpenAI API 兼容，支持时间戳和元数据"""
    
    content: str
    role: MessageRole
    # 动态默认时间戳（创建时自动设为当前时间）
    timestamp: datetime = Field(default_factory=datetime.now)
    # 可选的元数据（默认为空字典）
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # 可选：支持 OpenAI 的 tool 角色相关字段
    tool_call_id: Optional[str] = None
    name: Optional[str] = None   # 可用于 function/tool 的名称
    
    model_config = {
        "extra": "forbid",   # 禁止传入未声明的字段
        "str_strip_whitespace": True,  # 自动去除 content 首尾空白（可选）
    }

    # 加一个 __init__ 兼容位置参数
    def __init__(self, content: str = "", role: str = "user", **kwargs):
        super().__init__(content=content, role=role, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为 OpenAI API 所需的字典（仅包含 role 和 content）"""
        return {"role": self.role, "content": self.content} 
    
    def __str__(self) -> str:
        """友好的打印格式"""
        return f"[{self.role}] {self.content}"