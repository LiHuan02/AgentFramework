import hashlib
from typing import Dict, Any, List
from ..base import Tool, ToolParameter


class HashTool(Tool):
    """真实的哈希计算工具，支持 MD5、SHA1、SHA256、SHA512 等"""

    def __init__(self):
        super().__init__(
            name="hash",
            description="计算输入文本的哈希值（支持 MD5、SHA1、SHA256、SHA512）",
            expandable=False
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="text",
                type="string",
                description="要计算哈希的原始文本",
                required=True
            ),
            ToolParameter(
                name="algorithm",
                type="string",
                description="哈希算法，可选 md5 / sha1 / sha256 / sha512，默认 sha256",
                required=False,
                default="sha256"
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        text = parameters["text"]
        algorithm = parameters.get("algorithm", "sha256").strip().lower()

        # 支持的算法映射
        algo_map = {
            "md5": hashlib.md5,
            "sha1": hashlib.sha1,
            "sha256": hashlib.sha256,
            "sha512": hashlib.sha512,
        }

        if algorithm not in algo_map:
            supported = ", ".join(algo_map.keys())
            return f"错误：不支持的哈希算法 '{algorithm}'，可选：{supported}"

        # 计算哈希
        hash_obj = algo_map[algorithm]()
        hash_obj.update(text.encode("utf-8"))
        hex_digest = hash_obj.hexdigest()

        return f"{algorithm.upper()} 哈希结果：{hex_digest}"
