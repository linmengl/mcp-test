import datetime
import requests
import json
import re


def get_weather(location: str, date: str) -> str:
    # 简化版：只支持 today / tomorrow
    city_coords = {
        "北京": (39.9042, 116.4074),
        "上海": (31.2304, 121.4737),
        "深圳": (22.5431, 114.0579)
    }

    if location not in city_coords:
        return f"暂不支持该城市：{location}"

    lat, lon = city_coords[location]

    target_day = datetime.date.today()
    if "明天" in date:
        target_day += datetime.timedelta(days=1)

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min&timezone=Asia%2FShanghai"

    resp = requests.get(url).json()
    max_temp = resp['daily']['temperature_2m_max'][0]
    min_temp = resp['daily']['temperature_2m_min'][0]

    return f"{date} {location}的天气：最高气温 {max_temp}°C，最低气温 {min_temp}°C"

def get_stock_price(symbol: str) -> str:
    symbol_map = {
        "苹果": "aapl",
        "腾讯": "00700.hk",
        "阿里": "baba",
        "茅台": "sh600519",
    }

    code = symbol_map.get(symbol.lower(), symbol.lower())

    url = f"https://api.uu.ee/finance/quote?symbol={code}"  # 这是第三方封装的新浪财经接口
    try:
        resp = requests.get(url, timeout=5).json()
        name = resp['data']['name']
        price = resp['data']['price']
        change = resp['data']['change_percent']
        return f"{name} 当前价格为 {price} 元，涨跌幅为 {change}"
    except:
        return f"无法获取股票信息：{symbol}"

functions = {
    "get_weather": get_weather,
    "get_stock_price": get_stock_price
}

def execute_function(response_json: str) -> str:
    try:
        data = json.loads(response_json)
        func_name = data["name"]
        args = data["arguments"]
        func = functions.get(func_name)
        if func:
            return func(**args)
        else:
            return f"未找到函数：{func_name}"
    except Exception as e:
        return f"函数调用失败：{e}"

def call_ollama_for_function_call(user_input):
    prompt = f"""
你是一个 AI 助手，可以调用以下函数：
1. get_weather(location: str, date: str)：获取某城市某日的天气信息
2. get_stock_price(symbol: str)：获取某股票的当前价格（支持中文或代码）

请根据用户的输入，输出如下 JSON 结构：
{{
  "name": "函数名",
  "arguments": {{
    "参数名1": "参数值1",
    ...
  }}
}}

用户输入：{user_input}
请只返回 JSON，不要输出解释。只返回json
"""
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "deepseek-r1:7b",
        "prompt": prompt,
        "stream": False
    })

    text = response.json()["response"]

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            print("-=----------------")
            print(match.group(0))
            print("-=----------------")
            return str(match.group(0))
        except json.JSONDecodeError as e:
            print("JSON 解析失败，内容如下：", match.group(0))
            raise e
    else:
        raise ValueError("未能从模型输出中提取出 JSON。原始内容：" + text)


def generate_final_answer(user_input: str, function_result: str) -> str:
    messages = [
        {"role": "system", "content": "你是一个助理，需要将函数结果转为自然语言回复用户。"},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": f"(你查询得到了如下结果：{function_result}) 请自然回答用户。"}
    ]

    resp = requests.post("http://localhost:11434/api/chat", json={
        "model": "deepseek-r1:7b",
        "messages": messages,
        "stream": False
    })
    return resp.json()["message"]["content"]

def run_function_calling_pipeline(user_input):
    print(f"\n🧾 用户输入：{user_input}")
    response_json = call_ollama_for_function_call(user_input)
    print(f"\n🧠 模型输出的函数调用：{response_json}")

    function_result = execute_function(response_json)
    print(f"\n🔧 本地执行结果：{function_result}")

    final = generate_final_answer(user_input, function_result)
    print(f"\n🗣️ 最终回答：{final}")

if __name__ == "__main__":
    run_function_calling_pipeline("查一下明天北京的天气")