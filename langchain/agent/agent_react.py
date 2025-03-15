

'''
增加agents的react能力，可以理解为  推理+操作
'''
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, initialize_agent,AgentType


llm = ChatOpenAI(model='gpt-4',temperature=0,max_tokens=1000)


tools = load_tools(['serpapi','llm-math'],llm=llm)

agent = initialize_agent(tools,llm,agent= AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,verbose=True)

agent.invoke({'姚明的老婆的身高是多少？'})