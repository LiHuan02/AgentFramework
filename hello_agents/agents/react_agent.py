"""ReAct Agent实现 - 推理与行动结合的智能体"""

import re
from typing import Optional, List, Tuple, Union, Iterator
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry

# 默认ReAct提示词模板
DEFAULT_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤（即每次只输出一对Thought-Action）：

Thought: 分析问题，确定需要什么信息，制定研究策略。
Action: 选择合适的工具获取信息，格式为以下之一：
- `{{tool_name}}[{{tool_input}}]`：调用工具获取信息。
- `Finish[研究结论]`：当你有足够信息得出结论时。

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循：工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动："""

class ReActAgent(Agent):
    """
    ReAct (Reasoning and Acting) Agent
    
    结合推理和行动的智能体，能够：
    1. 分析问题并制定行动计划
    2. 调用外部工具获取信息
    3. 基于观察结果进行推理
    4. 迭代执行直到得出最终答案
    
    这是一个经典的Agent范式，特别适合需要外部信息的任务。
    """
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None
    ):
        """
        初始化ReActAgent

        Args:
            name: Agent名称
            llm: LLM实例
            tool_registry: 工具注册表（可选，如果不提供则创建空的工具注册表）
            system_prompt: 系统提示词
            config: 配置对象
            max_steps: 最大执行步数
            custom_prompt: 自定义提示词模板
        """
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_steps = max_steps
        self.current_history: List[str] = []
        # 设置提示词模板：用户自定义优先，否则使用默认模板
        self.prompt_template = custom_prompt or DEFAULT_REACT_PROMPT

    def add_tool(self, tool, auto_expand: bool = True) -> None:
        """
        添加工具到工具注册表（便利方法）

        Args:
            tool: 工具实例
            auto_expand: 是否自动展开可展开的工具（默认True）
        """
        # 直接使用 ToolRegistry 的 register_tool，它会自动处理工具展开
        self.tool_registry.register_tool(tool, auto_expand=auto_expand)

    def remove_tool(self, tool_name: str) -> bool:
        """移除工具（便利方法）"""
        return self.tool_registry.unregister(tool_name)

    def list_tools(self) -> list:
        """列出所有可用工具"""
        return self.tool_registry.list_tools()

    def _build_prompt(self, question: str) -> str:
        """构建ReAct提示词"""
        tools_desc = self.tool_registry.get_tools_description()
        history_str = "\n".join(self.current_history)
        return self.prompt_template.format(
            tools=tools_desc,
            question=question,
            history=history_str
        )

    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """解析LLM输出，提取思考和行动"""
        # 使用海象运算符简化正则匹配
        thought = (m.group(1).strip() if (m := re.search(r"Thought:\s*(.*)", text)) else None)
        action = (m.group(1).strip() if (m := re.search(r"Action:\s*(.*)", text)) else None)
        return thought, action

    def _parse_action(self, action_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析行动文本，提取工具名称和输入
        同时适用于普通工具调用 tool_name[input] 和 Finish[结论]
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        return (match.group(1), match.group(2)) if match else (None, None)

    @staticmethod
    def _simulate_stream(text: str, chunk_size: int = 2) -> Iterator[str]:
        """将完整文本分块返回，模拟流式输出效果"""
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    def _execute_react_loop(self, input_text: str, **kwargs) -> str:
        """
        执行ReAct推理循环，返回最终答案
        中间步骤（思考、行动、观察）通过print实时输出
        """
        self.current_history = []
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        for step in range(1, self.max_steps + 1):
            print(f"\n--- 第 {step} 步 ---")

            # 构建提示词并调用LLM
            prompt = self._build_prompt(input_text)
            response_text = self.llm.invoke([{"role": "user", "content": prompt}], **kwargs)

            if not response_text:
                print("❌ 错误：LLM未能返回有效响应。")
                break

            # 解析输出
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"🤔 思考: {thought}")

            if not action:
                print("⚠️ 警告：未能解析出有效的Action，流程终止。")
                break

            # 解析行动：统一处理工具调用和Finish
            tool_name, tool_input = self._parse_action(action)

            if tool_name == "Finish":
                print(f"🎉 最终答案: {tool_input}")
                return tool_input

            # 执行工具调用
            if not tool_name or tool_input is None:
                self.current_history.append("Observation: 无效的Action格式，请检查。")
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            observation = self.tool_registry.execute_tool(tool_name, tool_input)
            print(f"👀 观察: {observation}")

            # 更新历史
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

        # 超过最大步数
        print("⏰ 已达到最大步数，流程终止。")
        return "抱歉，我无法在限定步数内完成这个任务。"

    def run(self, input_text: str, stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """
        运行ReAct Agent

        ReAct循环期间，中间步骤（思考、行动、观察）通过print实时输出。
        stream参数仅控制最终答案的返回方式。

        Args:
            input_text: 用户问题
            stream: 是否以流式形式返回最终答案
            **kwargs: 其他参数

        Returns:
            最终答案（字符串或字符串迭代器）
        """
        # 执行推理循环
        final_answer = self._execute_react_loop(input_text, **kwargs)

        # 保存到历史记录
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=final_answer, role="assistant"))

        return self._simulate_stream(final_answer) if stream else final_answer
