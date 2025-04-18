from typing import List, Dict, Optional

import util
from logger_util import setup_logger
from prompt_manager import PromptManager
from tool_call_manager import ToolCallManager
from util import ollama_chat

logger = setup_logger("AgentRunner")


class AgentRunner:
    def __init__(self, session, behavior_mode="auto"):
        self.session = session
        self.prompt_mgr = PromptManager()
        self.behavior_mode = behavior_mode
        self.tool_parser = ToolCallManager()

    async def run(self, query: str) -> str:
        self.prompt_mgr.reset()
        self.prompt_mgr.add_system(await self._build_system_prompt())
        self.prompt_mgr.add_user(query)

        while True:
            messages = self.prompt_mgr.get_messages()
            logger.debug("🔄 Sending messages to model.messages: %s", messages)
            llm_response = await ollama_chat(messages)
            llm_content = llm_response["message"]["content"]
            logger.debug("Model response: %s", llm_content)

            # 允许中途切换模式
            if mode := self._check_dynamic_mode_switch(llm_content):
                self.behavior_mode = mode
                logger.info("🔁 Behavior mode switched to: %s", mode)
                continue

            if "Final Answer:" in llm_content:
                self.prompt_mgr.add_assistant_message(llm_content)
                return llm_content.split("Final Answer:", 1)[1].strip()

            tool_calls = self.tool_parser.parse_tool_calls(llm_content)
            logger.info("解析需要调用的工具.tool_calls: %s", tool_calls)
            self.prompt_mgr.add_assistant_message(llm_content)

            if not tool_calls:
                logger.info("⚠️ 无法识别工具调用，返回模型回复")
                return llm_content  # fallback to LLM response

            # tool_outputs = []
            for call in tool_calls:
                try:
                    result = await self.session.call_tool(call.get("tool"), call.get("arguments"))
                    tool_output = parse_tool_result(result)
                    logger.info("工具返回结果.工具名: %s,result: %s", call.get("tool"), tool_output)
                    self.prompt_mgr.add_tool_result(tool_output, call.get("tool"))
                    # tool_outputs.append({"role": call.get("tool"), "result": tool_output, "tool_call_id": str(hash(call.get("tool")))})
                except Exception as e:
                    error = {"error": str(e)}
                    self.prompt_mgr.add_tool_result(error, call.get("tool"))
                    logger.info("工具调用异常.工具名: %s,result: %s", call.get("tool"), error)
                    # tool_outputs.append({"tool": call["tool"], "result": error})

            # logger.info("🔧 工具调用结果", tool_outputs)

    async def _get_tools(self) -> List[Dict]:
        tool_resp = await self.session.list_tools()
        return [{
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema
        } for t in tool_resp.tools]

    def _build_system_prompt(self, tools_prompt: str) -> str:
        base = (
            f"你是一个聪明、可靠、遵循规则的 AI 助手，可以访问如下工具：\n"
            f"{tools_prompt}\n\n---\n"
            "📌 使用规则：\n"

            "✅ 1. 如果你可以直接回答用户的问题（不需要调用工具），请使用以下格式回复：\n"
            "Final Answer: [你的最终回答]\n"
            "示例：\n"
            "Final Answer: 今天天气晴朗，气温 25°C。\n"
            "🔧 2. 如果你需要调用工具，请严格使用以下 JSON 格式回复（不要输出其他内容）：\n"

            "```json\n"
            "{{\n"
            "  \"tool\": \"工具名\",\n"
            "  \"arguments\": {{\n"
            "    \"参数1\": \"值\",\n"
            "    \"参数2\": \"值\"\n"
            "  }}\n"
            "}}\n```\n"
        )

        if self.behavior_mode == "tool-first":
            base += "\n💡 当前为 Tool-First 模式：请优先判断是否可调用工具解决问题，再考虑直接回答。"
        elif self.behavior_mode == "llm-first":
            base += "\n🧠 当前为 LLM-First 模式：请尽量直接回答问题，仅在必要时调用工具。"
        else:
            base += "\n⚙️ 当前为 Auto 模式：你可以自行判断是否使用工具。"

        return base

    def _check_dynamic_mode_switch(self, response: str) -> Optional[str]:
        lowered = response.lower()
        if "mode:tool-first" in lowered:
            return "tool-first"
        elif "mode:llm-first" in lowered:
            return "llm-first"
        elif "mode:auto" in lowered:
            return "auto"
        return None

    async def _build_system_prompt(self) -> str:
        tool_response = await self.session.list_tools()
        tools = [{
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema
        } for t in tool_response.tools]
        return util.build_system_prompt(tools)


def parse_tool_result(result) -> str:
    """
    提取 MCP 工具返回结果中的文本内容。

    Args:
        result: 工具调用返回结果（一般含有 .content, .isError 等属性）

    Returns:
        格式化后的纯文本字符串。如果出错，返回错误信息。
    """
    if getattr(result, "isError", False):
        return "[❌ 工具调用失败]"

    content_blocks = getattr(result, "content", [])
    if not content_blocks:
        return "[⚠️ 工具未返回任何内容]"

    texts = []
    for block in content_blocks:
        text = getattr(block, "text", None)
        if text:
            texts.append(text.strip())

    return "\n\n".join(texts) if texts else "[⚠️ 工具返回内容无法解析]"
