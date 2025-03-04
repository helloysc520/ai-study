import requests
import time


#定义请求url
url = "http://127.0.0.1:5000/v1/main_server/"
data = {'uid': 'AI-6-202104','text': '公共英语(PETS)写作中常见的逻辑词汇汇总'}

start_time = time.time()
res = requests.post(url=url,json=data,headers={'Content-Type': 'application/json'})

cost_time = time.time() - start_time

#打印返回的结果
print('输入文本：',data['text'])
print('分类结果:',res.text)

print('单条样本预测耗时：',cost_time * 1000,'ms')