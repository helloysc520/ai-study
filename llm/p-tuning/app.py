import json

import torch
from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn,json,datetime
from transformers import AutoTokenizer,AutoModel,AutoConfig

'''
使用fastapi部署
'''

app = FastAPI()

#允许所有请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.post('/')
async def create_item(request: Request):
    global model,tokenizer

    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')

    response,history = model.chat(tokenizer,
                                  prompt,
                                  history=history,
                                  max_length= max_length if max_length else 2048,
                                  top_p=top_p if top_p else 0.7,
                                  temperature=temperature if temperature else 0.95

    )

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        'response': response,
        'history': history,
        'status': 200,
        'time': time
    }
    log = "[" +time + "]" + '",prompt:"' + prompt + '",response:"' + repr(response) + '"'

    print(log)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return answer

if __name__ == '__main__':

    pre_seq_length = 300
    model_path = ''
    checkpoint_path = '' ##ptuning微调后的前缀结果

    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path,trust_remote_code=True,pre_seq_length= pre_seq_length)
    model = AutoModel.from_pretrained(model_path,trust_remote_code=True,config=config)

    prefix_state_dict = torch.load(checkpoint_path,'pytorch_model.bin')

    new_prefix_state_dict = {}

    for k,v in prefix_state_dict.items():
        if k.startswith('transformer.prefix_encoder.'):

            new_prefix_state_dict[k[len('transformer.prefix_encoder.'):]] = v

    model.transformer.pre_encoder.load_state_dict(new_prefix_state_dict)

    ##量化
    model = model.quantize(4)
    model = model.cuda()

    model.eval()

    uvicorn.run(app,host='0.0.0.0',port=8103,workers=1)
