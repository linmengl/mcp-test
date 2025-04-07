from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# 定义一个模板
prompt = PromptTemplate(
    input_variables=["name"],
    template="你好, {name}！今天怎么样？"
)

# 使用 OpenAI 模型
llm = OpenAI(temperature=0.7)

# 创建链
chain = LLMChain(llm=llm, prompt=prompt)

# 执行链
result = chain.run("Alice")
print(result)