from openai import OpenAI
import tiktoken

#定义函数 num_tokens_from_messages
def num_tokens_from_messages(messages,model='gpt-3.5-turbo-0613'):
    #尝试获取模型的编码
    try:
        encoding = tiktoken.encoding_for_model(model)

    except KeyError:
        #如果没找到，使用cl100k_base编码并给出警告
        print('Waring:model not found. Using cl100k_base encoding.')

        encoding = tiktoken.get_encoding('cl100k_base')



    #针对不同模型设置数量
    if  model in {

        'gpt-3.5-turbo-0613',
        'gpt-3.5-turbo-16k-0613',
        'gpt-4-0314',
        'gpt-4-32k-0314',
        'gpt-4-0613',
        'gpt-4-32k-0613'
    }:
        tokens_per_message = 3
        tokens_per_name = 1

    elif model == 'gpt-3.5-turbo-0301':

        tokens_per_message = 4
        tokens_per_name = -1

    elif 'gpt-3.5-turbo' in model:

        print('Waring:gpt-3.5-turbo may update over time.Returning num tokens assuming gpt-3.5-turbo-0613')

        return num_tokens_from_messages(messages,model='gpt-3.5-turbo-0613')

    elif 'gpt-4' in model:

        print('Waring:gpt-4 may update over time.Returning num tokens assuming gpt-4-0613')
        return num_tokens_from_messages(messages,model='gpt-4-0613')

    elif  model in {
        'davinci',
        'curie',
        'babbage',
        'ada'
    }:
        print('Waring:gpt-3 related model is used. Returing num tokens assuming gpt2.')
        encoding = tiktoken.get_encoding('gpt2')
        num_tokens = 0

        for message in messages:
            for key,value in message.items():
                if key == 'content':
                    num_tokens += len(encoding.encode(value))

        return num_tokens

    else:
        #对于没有实现的模型，抛出未实现错误
        raise NotImplementedError(

            f"'xxx'"
        )

    num_tokens = 0
    for message in messages:
        for key, value in message.items():

            num_tokens += len(encoding.encode(value))
            if key == 'name':
                num_tokens += tokens_per_name

    num_tokens += 3
    return num_tokens

example_messages = [

    {
        'role':'system',
        'content':'You are a helpful,pattern-following assistant that translate corporate jargon into plain English.'
    },
    {
        'role':'system',
        'name': 'example_user',
        'content':'New synergies will help drive top-line growth.'
    }
]

client = OpenAI(api_key= 'sk.xxxxx')

for model in[
  'gpt-3.5-turbo',
    'gpt-4-0613',
    'gpt-4'
]:
    print(model)

    response = client.chat.completions.create(
        model=model,
        messages=example_messages
    )

    print(f'{response.usage.prompt_tokens} prompt tokens counted by the OpenAI API.')
