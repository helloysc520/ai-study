from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(temperature=0,max_tokens=1000)

conversation = ConversationChain(llm=llm ,verbose=True,memory= ConversationBufferMemory)

conversation.predict(input='嗨，你好啊')

res = conversation.predict(input= '我很好，正在和AI进行对话呢!')
print(res)

res = conversation.predict(input= '给我讲讲关于你的事情吧.')

print(res)



'''
为记忆模块提供窗口机制
'''

conversation_with_summary = ConversationChain(

    llm= llm,
    verbose=True,
    memory= ConversationBufferWindowMemory(k=2),
)

res = conversation_with_summary.predict(input= '嗨，最近一切可好？')
print(res)
print('---------------------')

res = conversation_with_summary.predict(input= '出了什么问题吗？')
print(res)
print('---------------------')

res = conversation_with_summary.predict(input= '现在情况好转了吗?')
print(res)
print('---------------------')

#注意，第一轮对话的信息已经消失不见了.
res = conversation_with_summary.predict(input='那是怎么解决的?')
print(res)

