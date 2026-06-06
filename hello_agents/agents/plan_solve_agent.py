"""Plan and Solve Agent实现 - 分解规划与逐步执行的智能体"""

import re
import ast
from typing import Optional, List, Dict, Union, Iterator
from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message
from ..tools.registry import ToolRegistry

# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。

## 可用工具
{tools}

你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。
如果某个步骤需要使用工具，请在步骤描述中明确指出使用哪个工具。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

## 可用工具
{tools}

如果当前步骤需要使用工具，请使用以下格式调用工具：[TOOL_CALL:工具名:参数]
如果你不需要使用工具，直接输出当前步骤的答案即可。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""

class Planner:
    """规划器 - 负责将复杂问题分解为简单步骤"""

    def __init__(self, llm_client: HelloAgentsLLM, tool_registry: Optional[ToolRegistry] = None, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.prompt_template = prompt_template or DEFAULT_PLANNER_PROMPT

    def plan(self, question: str, **kwargs) -> List[str]:
        """
        生成执行计划

        Args:
            question: 要解决的问题
            **kwargs: LLM调用参数

        Returns:
            步骤列表
        """
        # 获取工具描述，如果没有工具则为"无"
        tools_desc = self.tool_registry.get_tools_description() if self.tool_registry else "无"
        
        prompt = self.prompt_template.format(question=question, tools=tools_desc)
        messages = [{"role": "user", "content": prompt}]

        print("\n--- 正在生成计划 ---")
        response_text = self.llm_client.invoke(messages, **kwargs) or ""
        print(f"✅ 计划已生成:\n{response_text}")

        try:
            # 提取Python代码块中的列表
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []

class Executor:
    """执行器 - 负责按计划逐步执行"""

    def __init__(self, llm_client: HelloAgentsLLM, tool_registry: Optional[ToolRegistry] = None, prompt_template: Optional[str] = None):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.prompt_template = prompt_template or DEFAULT_EXECUTOR_PROMPT

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        """
        按计划执行任务

        Args:
            question: 原始问题
            plan: 执行计划
            **kwargs: LLM调用参数

        Returns:
            最终答案
        """
        history = ""
        final_answer = ""
        tools_desc = self.tool_registry.get_tools_description() if self.tool_registry else "无"

        print("\n--- 正在执行计划 ---")
        for i, step in enumerate(plan, 1):
            print(f"\n-> 正在执行步骤 {i}/{len(plan)}: {step}")
            prompt = self.prompt_template.format(
                question=question,
                plan="\n".join([f"{idx+1}. {s}" for idx, s in enumerate(plan)]),
                tools=tools_desc,
                history=history if history else "无",
                current_step=step
            )
            messages = [{"role": "user", "content": prompt}]

            response_text = self.llm_client.invoke(messages, **kwargs) or ""
            
            # 检查并执行工具调用
            step_result = self._check_and_execute_tool(response_text)

            history += f"步骤 {i}: {step}\n结果: {step_result}\n\n"
            final_answer = step_result
            print(f"✅ 步骤 {i} 已完成，结果: {final_answer}")

        return final_answer

    def _check_and_execute_tool(self, response_text: str) -> str:
        """检查响应中是否包含工具调用，如果有则执行并返回结果"""
        # 匹配格式：[TOOL_CALL:工具名:参数]
        tool_call_match = re.search(r'\[TOOL_CALL:(\w+):(.+?)\]', response_text)
        
        if tool_call_match and self.tool_registry:
            tool_name = tool_call_match.group(1)
            tool_args = tool_call_match.group(2)
            print(f"  🔧 检测到工具调用: {tool_name}({tool_args})")
            
            observation = self.tool_registry.execute_tool(tool_name, tool_args)
            print(f"  👀 工具返回: {observation}")
            return str(observation)
            
        # 如果没有工具调用，直接返回LLM的文本响应
        return response_text

class PlanAndSolveAgent(Agent):
    """
    Plan and Solve Agent - 分解规划与逐步执行的智能体
    
    这个Agent能够：
    1. 将复杂问题分解为简单步骤
    2. 按照计划逐步执行
    3. 维护执行历史和上下文
    4. 得出最终答案
    
    特别适合多步骤推理、数学问题、复杂分析等任务。
    """
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None
    ):
        """
        初始化PlanAndSolveAgent

        Args:
            name: Agent名称
            llm: LLM实例
            tool_registry: 工具注册表（可选）
            system_prompt: 系统提示词
            config: 配置对象
            custom_prompts: 自定义提示词模板 {"planner": "", "executor": ""}
        """
        super().__init__(name, llm, system_prompt, config)
        
        # 初始化工具注册表
        self.tool_registry = tool_registry or ToolRegistry()

        # 设置提示词模板
        planner_prompt = custom_prompts.get("planner") if custom_prompts else None
        executor_prompt = custom_prompts.get("executor") if custom_prompts else None

        # 将工具注册表传递给规划器和执行器
        self.planner = Planner(self.llm, self.tool_registry, planner_prompt)
        self.executor = Executor(self.llm, self.tool_registry, executor_prompt)

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

    def run(self, input_text: str, stream: bool = False, **kwargs) -> Union[str, Iterator[str]]:
        """
        运行Plan and Solve Agent
        
        Args:
            input_text: 要解决的问题
            stream: 是否以流式形式返回最终答案
            **kwargs: 其他参数
            
        Returns:
            最终答案（字符串或字符串迭代器）
        """
        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")
        
        # 1. 生成计划
        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止。"
            print(f"\n--- 任务终止 ---\n{final_answer}")
            
            # 保存到历史记录 (兼容Pydantic V2)
            self.add_message(Message(content=input_text, role="user"))
            self.add_message(Message(content=final_answer, role="assistant"))
            
            return self._simulate_stream(final_answer) if stream else final_answer
        
        # 2. 执行计划
        final_answer = self.executor.execute(input_text, plan, **kwargs)
        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")
        
        # 保存到历史记录 (兼容Pydantic V2)
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=final_answer, role="assistant"))
        
        return self._simulate_stream(final_answer) if stream else final_answer
