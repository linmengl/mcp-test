import json
from typing import List,Dict

class PromptManager:
    def __init__(self):
        self.messages:List[Dict] = []

    def reset(self):
        self.messages = []

    def add_system(self, content: str):
        self.messages.append({"role": "system", "content": content})

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, result, tool_name: str):
        self.messages.append({
            # "role": "tool",
            "role": "user",
            # "tool_call_id": str(hash(tool_name)),
            # "content": smart_print(result)
            "content": f"[工具 `{tool_name}` 返回的结果如下，请生成最终自然语言回答]\n{result}"
        })

    def get_messages(self):
        return self.messages

import json

def smart_print(obj):
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        print(obj)