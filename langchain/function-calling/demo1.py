import openai
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential
from termcolor import colored
from openai import OpenAI
import os
os.environ['OPENAI_API_KEY'] = ''
openai.api_key = ''

GPT_MODEL = 'gpt-3.5-turbo'
#直接访问openai的方式
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


#定义一个函数chat_completion_request，主要用于发送聊天补全请求到openai服务器
@retry(wait=wait_random_exponential(multiplier=1,max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages,functions=None,function_call=None,model=GPT_MODEL):

    headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + openai.api_key
               }

    #设定请求的json数据,
    json_data = {
        'model': model,
        'messages': messages
    }

    #如果传入了functions，将其加入到json_data中
    if functions is not None:
        json_data.update({'functions': functions})

    #如果传入了function_call,将其加入到json_data中

    if function_call is not None:
        json_data.update({'function_call': function_call})


    try:
        response = requests.post(
            'https://api.openai.com/v1/responses',
            headers=headers,
            json=json_data
        )
        return response

    except Exception as e:

        print('unable to generate chat completion response')
        print(f'Exception: {e}')
        return e


'''
定义 pretty_print_conversation，用于打印消息对话内容
'''
def pretty_print_conversation(messages):

    #为不同的角色设置不同的颜色
    rolo_to_color = {

        'system':'red',
        'user': 'green',
        'assistant':'blue',
        'function': 'magenta',
    }

    #遍历消息列表
    for message in messages:

        if message['role'] == 'system':
            print(colored(f"system:{message['content']}\n",rolo_to_color[message['role']]))

        elif message['role'] == 'user':
            print(colored(f"user:{message['content']}\n", rolo_to_color[message['role']]))

        elif message['role'] == 'assistant' and message.get('function_call'):
            print(colored(f"assistant[function_call]:{message['function_call']}\n", rolo_to_color[message['role']]))

        elif message['role'] == 'assistant' and not message.get('function_call'):
            print(colored(f"assistant[content]:{message['content']}\n", rolo_to_color[message['role']]))

        elif message['role'] == 'function':

            print(colored(f"function({message['name']}):{message['content']}\n", rolo_to_color[message['role']]))




functions = [

    #第一个字典定义了一个名为get_current_weather的功能
    {
        'name' : 'get_current_weather',   #功能名称
        'description' : 'Get the current weather', # 功能描述
        #定义该功能需要的参数
        'parameters' : {
            'type' : 'object',
            'properties': {
                'location' : {
                    'type' : 'string',
                    'description' : '城市与国家，北京，中国'
                },
                'format':{
                    'type' : 'string',
                    'enum': ['celsius','fahrenheit'],
                    'description': '使用的温度单位'
                }
            },
            'required': ['location', 'format']
        }

    },

    #第二个字典定义了一个名为get_n_day_weather_forecast的功能
    {
        'name': 'get_n_day_weather_forecast',
        'description' : 'Get a n-day weather forecast',
        'parameters' : {
            'type' : 'object',
            'properties': {
                'location' : {
                    'type' : 'string',
                    'description': '城市与国家，北京，中国'
                },
                'format':{
                    'type' : 'string',
                    'enum': ['celsius','fahrenheit'],
                    'description': '使用的温度单位'
                },
                'num_days':{
                    'type' : 'integer',
                    'description': '预测的天数'
                }
            },
            'required': ['location', 'format', 'num_days']
        },

    }
]

messages = []
messages.append({

    'role' : 'system',
    'content': '不要对函数中的任何值做假设，如果用户的请求是模糊不清的，请要求用户给出准确信息.'
})

messages.append({
    'role' : 'user',
    'content': '今天的天气怎么样？'
})

chat_response = chat_completion_request(messages,functions=functions)

#解析返回的json数据，获取助手的回复消息
assistant_message = chat_response.json()['choices'][0]['message']
print(type(assistant_message))

messages.append(assistant_message)

pretty_print_conversation(messages)



