"""Reflection Agent实现 - 自我反思与迭代优化的智能体"""

import re
from typing import Optional, List, Dict, Any, Union, Iterator, Tuple
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry

# 默认提示词模板
DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务：

## 可用工具
{tools}

如果需要使用工具来辅助完成任务（例如执行代码、搜索信息），请使用以下格式调用工具：[TOOL_CALL:工具名:参数]

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间：

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
特别注意：如果回答中包含了工具调用结果，请检查工具调用是否正确，结果是否符合预期。
如果回答已经够好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答：

## 可用工具
{tools}

如果需要使用工具来辅助完成任务（例如执行代码、搜索信息），请使用以下格式调用工具：[TOOL_CALL:工具名:参数]

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
"""
}

class Memory:
    """
    简单的短期记忆模块，用于存储智能体的行动与反思轨迹。
    """
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """向记忆中添加一条新记录"""
        self.records.append({"type": record_type, "content": content})
        print(f"📝 记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        """将所有记忆记录格式化为一个连贯的字符串文本"""
        trajectory = ""
        for record in self.records:
            if record['type'] == 'execution':
                trajectory += f"--- 上一轮尝试 ---\n{record['content']}\n\n"
            elif record['type'] == 'reflection':
                trajectory += f"--- 评审员反馈 ---\n{record['content']}\n\n"
        return trajectory.strip()

    def get_last_execution(self) -> str:
        """获取最近一次的执行结果"""
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return ""


class ReflectionAgent(Agent):
    """
    Reflection Agent - 自我反思与迭代优化的智能体

    这个Agent能够：
    1. 执行初始任务（支持调用工具）
    2. 对结果进行自我反思
    3. 根据反思结果进行优化（支持调用工具）
    4. 迭代改进直到满意

    特别适合代码生成、文档写作、分析报告等需要迭代优化的任务。
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 5,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        初始化ReflectionAgent

        Args:
            name: Agent名称
            llm: LLM实例
            tool_registry: 工具注册表（可选）
            system_prompt: 系统提示词
            config: 配置对象
            max_iterations: 最大迭代次数
            custom_prompts: 自定义提示词模板 {"initial": "", "reflect": "", "refine": ""}
        """
        super().__init__(name, llm, system_prompt, config)
        
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_iterations = max_iterations
        self.memory = Memory()

        # 设置提示词模板：用户自定义优先，否则使用默认模板
        self.prompts = custom_prompts if custom_prompts else DEFAULT_PROMPTS

    def add_tool(self, tool, auto_expand: bool = True) -> None:
        """添加工具到工具注册表"""
        self.tool_registry.register_tool(tool, auto_expand=auto_expand)

    def remove_tool(self, tool_name: str) -> bool:
        """移除工具"""
        return self.tool_registry.unregister(tool_name)

    def list_tools(self) -> list:
        """列出所有可用工具"""
        return self.tool_registry.list_tools()

    @staticmethod
    def _simulate_stream(text: str, chunk_size: int = 2) -> Iterator[str]:
        """将完整文本分块返回，模拟流式输出效果"""
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    def _check_and_execute_tool(self, response_text: str) -> Tuple[str, bool]:
        """
        检查响应中是否包含工具调用，如果有则执行并返回结果
        Returns:
            Tuple[执行结果文本, 是否调用了工具]
        """
        tool_call_match = re.search(r'\[TOOL_CALL:(\w+):(.+?)\]', response_text)
        
        if tool_call_match and self.tool_registry:
            tool_name = tool_call_match.group(1)
            tool_args = tool_call_match.group(2)
            print(f"  🔧 检测到工具调用: {tool_name}({tool_args})")
            
            observation = self.tool_registry.execute_tool(tool_name, tool_args)
            print(f"  👀 工具返回: {observation}")
            
            # 将工具结果拼接到原始回答中，方便后续反思评审
            full_content = f"{response_text}\n[工具执行结果]: {observation}"
            return full_content, True
            
        return response_text, False

    def _get_tools_description(self) -> str:
        """获取工具描述文本"""
        return self.tool_registry.get_tools_description() if self.tool_registry and self.tool_registry.list_tools() else "无"

    def _get_llm_response(self, prompt: str, **kwargs) -> str:
        """调用LLM并获取完整响应"""
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, **kwargs) or ""

    def run(self, input_text: str, stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """
        运行Reflection Agent

        Args:
            input_text: 任务描述
            stream: 是否以流式形式返回最终答案
            **kwargs: 其他参数

        Returns:
            最终优化后的结果（字符串或字符串迭代器）
        """
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")

        # 重置记忆
        self.memory = Memory()
        tools_desc = self._get_tools_description()

        # 1. 初始执行
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = self.prompts["initial"].format(task=input_text, tools=tools_desc)
        initial_response = self._get_llm_response(initial_prompt, **kwargs)
        
        # 执行工具并保存记录
        final_initial_content, _ = self._check_and_execute_tool(initial_response)
        print(final_initial_content)
        self.memory.add_record("execution", final_initial_content)

        # 2. 迭代循环：反思与优化
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")

            # a. 反思 (反思阶段不需要调用工具，只需评审已有内容)
            print("\n-> 正在进行反思...")
            last_result = self.memory.get_last_execution()
            reflect_prompt = self.prompts["reflect"].format(
                task=input_text,
                content=last_result
            )
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            print(feedback)
            self.memory.add_record("reflection", feedback)

            # b. 检查是否需要停止
            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                print("\n✅ 反思认为结果已无需改进，任务完成。")
                break

            # c. 优化
            print("\n-> 正在进行优化...")
            refine_prompt = self.prompts["refine"].format(
                task=input_text,
                tools=tools_desc,
                last_attempt=last_result,
                feedback=feedback
            )
            refined_response = self._get_llm_response(refine_prompt, **kwargs)
            
            # 执行工具并保存记录
            final_refined_content, _ = self._check_and_execute_tool(refined_response)
            print(final_refined_content)
            self.memory.add_record("execution", final_refined_content)

        final_result = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终结果:\n{final_result}")

        # 保存到历史记录 (兼容Pydantic V2)
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=final_result, role="assistant"))

        return self._simulate_stream(final_result) if stream else final_result
