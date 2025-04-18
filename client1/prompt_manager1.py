import json
from typing import List, Dict


class PromptManager:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.dialog: List[Dict[str, str]] = []

    def add_user_message(self, content: str):
        self.dialog.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.dialog.append({"role": "assistant", "content": content})

    def add_tool_call(self, tool_name: str, arguments: dict):
        content = json.dumps({
            "tool": tool_name,
            "arguments": arguments
        }, ensure_ascii=False)
        self.dialog.append({"role": "assistant", "content": content})  # 兼容 ChatML 模型习惯

    def add_tool_result(self, result: dict):
        self.dialog.append({
            "role": "tool",
            "content": json.dumps(result, ensure_ascii=False)
        })

    def get_messages(self) -> List[Dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}] + self.dialog

    def build_chatml_prompt(self) -> str:
        """
        把 messages 转换为 deepseek-r1 可识别的 ChatML prompt 字符串
        """
        all_messages = self.get_messages()
        return "\n".join([f"<|{m['role']}|>\n{m['content'].strip()}" for m in all_messages])

    def reset(self):
        self.dialog = []