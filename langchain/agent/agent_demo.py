import openai
import os
from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.agents import create_openapi_agent,AgentExecutor
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

os.environ['OPENAI_API_KEY']='<KEY>'
openai.api_key = ''

#定义大模型
llm = ChatOpenAI(temperature=0,max_tokens=1000)

@tool
def get_word_length(word:str) -> int:
    """ returns the length of a word """
    return len(word)


#定义工具tools
tools = [get_word_length]


#定义对话模板
prompt = ChatPromptTemplate.from_messages(
    [
        ('system','You are a helpful assistant,but bad at calculating lengths of words'),

        MessagesPlaceholder('chat_history',optional=True),
        ('human','{input}'),
        MessagesPlaceholder('agent_scratchpad'),
    ]
)

#实例化agent
agent = create_openapi_agent(llm,tools,prompt)

#实例化agent执行器
agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)

res = agent_executor.invoke({'input': 'how many letters in the word educa?'})



"""
增加agent的多轮对话记忆能力
"""
from langchain.memory import ConversationBufferMemory
MEMORY_KEY = 'chat_history'

memory = ConversationBufferMemory(memory_key=MEMORY_KEY,return_messages=True)

agent = create_openapi_agent(llm,tools,prompt)
agent_executor = AgentExecutor(agent=agent,tools=tools,memory=memory,verbose=True)
res = agent_executor.invoke({'input': 'how many letters in the word educa?'})
res = agent_executor.invoke({'input': 'is that a real word?'})



