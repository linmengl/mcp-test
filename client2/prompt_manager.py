import json

class PromptManager:
    def __init__(self):
        self.messages = []

    def reset(self):
        self.messages = []

    def add_system(self, content: str):
        self.messages.append({"role": "system", "content": content})

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, result: dict, tool_name: str):
        self.messages.append({
            "role": "tool",
            "tool_call_id": str(hash(tool_name)),
            "content": json.dumps(result)
        })

    def get_messages(self):
        return self.messages