"""简单Agent实现 - 基于OpenAI原生API"""

from typing import Optional, Iterator, TYPE_CHECKING, Union
import re
import json

from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry

class SimpleAgent(Agent):
    """简单的对话Agent，支持可选的工具调用"""
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True
    ):
        """
        初始化SimpleAgent
        
        Args:
            name: Agent名称
            llm: LLM实例
            system_prompt: 系统提示词
            config: 配置对象
            tool_registry: 工具注册表（可选，如果提供则启用工具调用）
            enable_tool_calling: 是否启用工具调用（只有在提供tool_registry时生效）
        """
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
    
    def _get_enhanced_system_prompt(self) -> str:
        """构建增强的系统提示词，包含工具信息"""
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"
        
        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt
        
        # 获取工具描述
        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "暂无可用工具":
            return base_prompt
        
        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题：\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式（parameters使用 `key=value` 格式，用逗号分隔）：\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n\n"

        tools_section += "### 参数格式示例\n"
        tools_section += "   示例1：`[TOOL_CALL:calculator_multiply:a=12,b=8]`\n"
        tools_section += "   示例2：`[TOOL_CALL:filesystem_read_file:path=README.md]`\n\n"
        tools_section += "   示例3：`[TOOL_CALL:search:query=Python编程]`\n\n"

        tools_section += "### 重要提示\n"
        tools_section += "- 参数名必须与工具定义的参数名完全匹配\n"
        tools_section += "- 数字参数直接写数字，不需要引号：`a=12` 而不是 `a=\"12\"`\n"
        tools_section += "- 文件路径等字符串参数直接写：`path=README.md`\n"
        tools_section += "- 工具调用结果会自动插入到对话中，然后你可以基于结果继续回答\n"

        return base_prompt + tools_section
    
    def _build_messages(self, input_text: str) -> list:
        """构建发送给LLM的消息列表"""
        messages = [{"role": "system", "content": self._get_enhanced_system_prompt()}]
        messages.extend({"role": msg.role, "content": msg.content} for msg in self._history)
        messages.append({"role": "user", "content": input_text})
        return messages

    def _parse_tool_calls(self, text: str) -> list:
        """解析文本中的工具调用"""
        pattern = r'\[TOOL_CALL:([^:]+):([^\]]+)\]'
        return [
            {
                'tool_name': match[0].strip(), 
                'parameters': match[1].strip(), 
                'original': f'[TOOL_CALL:{match[0]}:{match[1]}]'
            } 
            for match in re.findall(pattern, text)
        ]
    
    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        """执行工具调用"""
        if not self.tool_registry:
            return "❌ 错误：未配置工具注册表"

        try:
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                return f"❌ 错误：未找到工具 '{tool_name}'"

            param_dict = self._parse_tool_parameters(tool_name, parameters)
            result = tool.run(param_dict)
            return f"🔧 工具 {tool_name} 执行结果：\n{result}"

        except Exception as e:
            return f"❌ 工具调用失败：{str(e)}"

    def _parse_tool_parameters(self, tool_name: str, parameters: str) -> dict:
        """智能解析工具参数"""
        if parameters.strip().startswith('{'):
            try:
                param_dict = json.loads(parameters)
                return self._convert_parameter_types(tool_name, param_dict)
            except json.JSONDecodeError:
                pass

        if '=' in parameters:
            pairs = [p.split('=', 1) for p in parameters.split(',') if '=' in p]
            param_dict = {k.strip(): v.strip() for k, v in pairs}
            param_dict = self._convert_parameter_types(tool_name, param_dict)
            return self._infer_action(tool_name, param_dict)
        
        return self._infer_simple_parameters(tool_name, parameters)

    def _convert_parameter_types(self, tool_name: str, param_dict: dict) -> dict:
        """根据工具的参数定义转换参数类型"""
        if not self.tool_registry:
            return param_dict

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            return param_dict

        try:
            param_types = {param.name: param.type for param in tool.get_parameters()}
        except Exception:
            return param_dict

        converted_dict = {}
        for key, value in param_dict.items():
            param_type = param_types.get(key)
            if not param_type:
                converted_dict[key] = value
                continue
            
            try:
                if param_type in ('number', 'integer') and isinstance(value, str):
                    converted_dict[key] = float(value) if param_type == 'number' else int(value)
                elif param_type == 'boolean' and isinstance(value, str):
                    converted_dict[key] = value.lower() in ('true', '1', 'yes')
                else:
                    converted_dict[key] = value
            except (ValueError, TypeError):
                converted_dict[key] = value

        return converted_dict

    def _infer_action(self, tool_name: str, param_dict: dict) -> dict:
        """根据工具类型和参数推断action"""
        if tool_name == 'memory':
            if 'recall' in param_dict:
                param_dict.update(action='search', query=param_dict.pop('recall'))
            elif 'store' in param_dict:
                param_dict.update(action='add', content=param_dict.pop('store'))
            elif 'query' in param_dict:
                param_dict['action'] = 'search'
            elif 'content' in param_dict:
                param_dict['action'] = 'add'
        elif tool_name == 'rag':
            if 'search' in param_dict:
                param_dict.update(action='search', query=param_dict.pop('search'))
            elif 'query' in param_dict:
                param_dict['action'] = 'search'
            elif 'text' in param_dict:
                param_dict['action'] = 'add_text'
        return param_dict

    def _infer_simple_parameters(self, tool_name: str, parameters: str) -> dict:
        """为简单参数推断完整的参数字典"""
        if tool_name in ('rag', 'memory'):
            return {'action': 'search', 'query': parameters}
        return {'input': parameters}

    def _run_tool_loop(self, messages: list, max_tool_iterations: int, **kwargs) -> str:
        """执行工具调用循环，返回最终的纯文本响应"""
        current_iteration = 0
        while current_iteration < max_tool_iterations:
            response = self.llm.invoke(messages, **kwargs)
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                return response

            clean_response = response
            tool_results = []
            for call in tool_calls:
                result = self._execute_tool_call(call['tool_name'], call['parameters'])
                tool_results.append(result)
                clean_response = clean_response.replace(call['original'], "")

            messages.append({"role": "assistant", "content": clean_response})
            messages.append({"role": "user", "content": f"工具执行结果：\n{'\n\n'.join(tool_results)}\n\n请基于这些结果给出完整的回答。"})

            current_iteration += 1

        # 如果超过最大迭代次数，获取最后一次回答
        return self.llm.invoke(messages, **kwargs)

    @staticmethod
    def _simulate_stream(text: str, chunk_size: int = 2) -> Iterator[str]:
        """将完整文本分块返回，模拟流式输出效果"""
        for i in range(0, len(text), chunk_size):
            yield text[i:i+chunk_size]

    def run(self, input_text: str, max_tool_iterations: int = 5, stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """
        运行SimpleAgent，支持可选的工具调用和流式输出
        
        Args:
            input_text: 用户输入
            max_tool_iterations: 最大工具调用迭代次数（仅在启用工具时有效）
            stream: 是否以流式形式返回响应
            **kwargs: 其他参数
            
        Returns:
            Agent响应（字符串或字符串迭代器）
        """
        messages = self._build_messages(input_text)
        
        # 1. 没有启用工具调用：可以直接使用真正的流式输出
        if not self.enable_tool_calling:
            if stream:
                def stream_generator():
                    full_response = ""
                    for chunk in self.llm.stream_invoke(messages, **kwargs):
                        full_response += chunk
                        yield chunk
                    self.add_message(Message(input_text, "user"))
                    self.add_message(Message(full_response, "assistant"))
                return stream_generator()
            else:
                response = self.llm.invoke(messages, **kwargs)
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(response, "assistant"))
                return response

        # 2. 启用了工具调用：由于需要完整文本解析 TOOL_CALL 标记，
        # 在工具调用循环期间无法逐字流式输出，因此先完整执行循环，再决定是否模拟流式输出
        final_response = self._run_tool_loop(messages, max_tool_iterations, **kwargs)
        
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))

        if stream:
            return self._simulate_stream(final_response)
        else:
            return final_response

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        流式运行Agent (兼容旧代码的便利方法)
        
        Args:
            input_text: 用户输入
            **kwargs: 其他参数
            
        Yields:
            Agent响应片段
        """
        return self.run(input_text, stream=True, **kwargs)

    def add_tool(self, tool, auto_expand: bool = True) -> None:
        """
        添加工具到Agent（便利方法）

        Args:
            tool: Tool对象
            auto_expand: 是否自动展开可展开的工具（默认True）

        如果工具是可展开的（expandable=True），会自动展开为多个独立工具
        """
        if not self.tool_registry:
            from ..tools.registry import ToolRegistry
            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True

        self.tool_registry.register_tool(tool, auto_expand=auto_expand)

    def remove_tool(self, tool_name: str) -> bool:
        """移除工具（便利方法）"""
        if self.tool_registry:
            self.tool_registry.unregister(tool_name)
            return True
        return False

    def list_tools(self) -> list:
        """列出所有可用工具"""
        return self.tool_registry.list_tools() if self.tool_registry else []

    def has_tools(self) -> bool:
        """检查是否有可用工具"""
        return self.enable_tool_calling and self.tool_registry is not None
