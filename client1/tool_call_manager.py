import json
import regex
from typing import List, Dict, Union


class ToolCallManager:

    @staticmethod
    def extract_json_blocks(text: str) -> List[str]:
        """
        从文本中提取所有合法 JSON 块
        """
        pattern = r'\{(?:[^{}]|(?R))*\}'  # 支持嵌套的 JSON 正则
        matches = regex.findall(pattern, text, regex.DOTALL)
        return matches

    @staticmethod
    def parse_tool_call_block(block: str) -> Union[Dict, None]:
        try:
            fixed = block.replace("'", '"')  # 单引号转双引号
            parsed = json.loads(fixed)

            # 校验是否为合法工具结构
            if "tool" in parsed and "arguments" in parsed:
                return parsed
        except Exception:
            return None

    @classmethod
    def parse_tool_calls(cls, content: str) -> List[Dict]:
        """
        尝试从模型输出中提取一个或多个工具调用结构
        返回格式：
        [
          {
            "tool": "xxx",
            "arguments": { ... }
          },
          ...
        ]
        """
        tool_calls = []

        # 尝试整体解析为 JSON 数组
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "tool" in item and "arguments" in item:
                        tool_calls.append(item)
                return tool_calls
        except Exception:
            pass

        # 如果不是数组，继续查找多个 JSON 块
        blocks = cls.extract_json_blocks(content)
        for block in blocks:
            parsed_call = cls.parse_tool_call_block(block)
            if parsed_call:
                tool_calls.append(parsed_call)

        return tool_calls