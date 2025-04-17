import json
import re


class ToolCallManager:
    @staticmethod
    def parse_tool_calls(content: str):
        """
        从 assistant 的输出中提取工具调用 JSON。
        支持一个或多个 tool 调用（这里只处理一个）。
        """
        try:
            return [json.loads(content)]
        except json.JSONDecodeError:
            # 尝试正则提取第一个 JSON 对象
            pattern = r'\{(?:[^{}]|\{[^{}]*\})*\}'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    fixed = match.group().replace("'", '"')
                    return [json.loads(fixed)]
                except json.JSONDecodeError:
                    return []
        return []