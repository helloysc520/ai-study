import time
import jieba
import fasttext

from flask import Flask
from flask import request

app= Flask(__name__)

import requests



#加载自定义的停用字典
jieba.load_userdict('E:\\AI_workspace\\ai-study\\toutiao\\data\\stopwords.txt')

model_save_path = 'E:\\AI_workspace\\ai-study\\toutiao\\baseline\\fast_text\\toutiao_fast_1740711508.bin'


#实例化已训练好的模型路径+名字
model = fasttext.load_model(model_save_path)
print('fasttext模型实例化完毕...')

@app.post('/v1/main_server')
def main_server():

    #接收来自请求方发送的服务字段
    uid = request.form['uid']
    text = request.form['text']

    input_text = ' '.join(jieba.lcut(text))

    #执行预测
    res = model.predict(input_text)
    predict_name = res[0][0]

    return predict_name