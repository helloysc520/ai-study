from langchain.schema import SystemMessage,HumanMessage,AIMessage
from langchain_openai import ChatOpenAI


'''
展示模型输入输出英文
'''
chat_model = ChatOpenAI(model_name = 'gpt-4')

messages = [

    SystemMessage(content='You are a helpful assistant.'),
    HumanMessage(content='Who won the world champion in 2022?'),
    AIMessage(content='The Golden Warriors won the World Champion in 2022.'),
    HumanMessage(content='Where was it played?')

]

chat_result = chat_model.invoke(messages)
print(chat_result)
print('----------------------')
print(type(chat_result))




'''
展示模型输入输出中文
'''

llm = ChatOpenAI(model_name = 'gpt-3.5-turbo',max_tokens=1024)
print(llm.max_tokens)

res = llm.invoke('讲3个给程序员听的笑话，要幽默诙谐的感觉!')

print(res)



