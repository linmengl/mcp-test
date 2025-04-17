from typing import Optional
import regex

class ToolCallParser:
    @staticmethod
    def parse(content: str) -> Optional[dict]:
        """
        尝试从 assistant 输出中提取 JSON 格式的工具调用请求
        格式示例：
        {
          "tool": "xxx",
          "arguments": { ... }
        }
        """
        import json
        import re

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 正则查找最外层 JSON 块（含嵌套）  pattern = r'\{(?:[^{}]|\{[^{}]*\})*\}'
            match = regex.search(r'\{(?:[^{}]|(?R))*\}', content, regex.DOTALL)
            if match:
                try:
                    fixed = match.group().replace("'", '"')  # 替换单引号
                    return json.loads(fixed)
                except Exception:
                    return None
        return None