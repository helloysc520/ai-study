import re
import json
import os
import time
import torch
from jieba import tokenize
from transformers import AutoModel,AutoTokenizer



# 提供一些示例让LLM进行few-shot
ie_examples = {
    '人物': [
        {
            'content': '岳云鹏, 本名岳龙刚, 1985年4月15日出生于河南省濮阳市南乐县, 中国内地相声, 影视男演员。',
            'answers': {
                '姓名': ['岳云鹏'],
                '性别': ['男'],
                '出生日期': ['1985年4月15日'],
                '出生地': ['河南省濮阳市南乐县'],
                '职业': ['相声演员', '影视演员'],
                '获得奖项': ['原文中未提及']
            }
        }
    ],
    '书籍': [
        {
            'content': '《三体》是刘慈欣创作的长篇科幻小说系列, 由《三体》《三体2:黑暗森林》《三体3:死神永生》组成, 第一部于2006年5月起在《科幻世界》杂志上连载, 第二部于2008年5月首次出版, 第三部则于2010年11月出版。',
            'answers': {
                '书名': ['《三体》'],
                '作者': ['刘慈欣'],
                '类型': ['长篇科幻小说'],
                '发行时间': ['2006年5月', '2008年5月', '2010年11月'],
                '定价': ['原文中未提及']
            }
        }
    ]
}

def init_prompts():

    pre_history = [
        {
            'content': f"你是一个信息抽取任务的专家，当我给你一段文本时，你需要帮我抽取句子中的三元组，并按照JSON格式输出，对于这段文本中没有的信息用['原文中未提及']来表示，多个输出值之间，用','分割",
            'role': 'system'
        }

    ]

    for type,example_list in ie_examples.items():

        for example in example_list:

            sentence = example['content']
            answers = json.dumps(example['answers'],ensure_ascii=False)

            new_data = {

                'content': f"对{sentence}这段文段进行抽取的结果是{answers}",
                'role': 'user'
            }
            pre_history.append(new_data)

            return pre_history


def clean_reponse(response):

    if '```json' in response:
        res = re.findall(r'```(.*?)```', response)

        if len(res) and res[0]:
            response = res[0]

        response.replace('、',',')

    try:
        return json.loads(response)

    except:
        return response

def predict(sentences,pre_history):

    for sentence in sentences:

        print('sentence:',sentence)
        start_time = time.time()
        response,_ = model.chat(tokenizer,sentence,history=pre_history)
        end_time = time.time()
        response = clean_reponse(response)

        print('response:',response)
        print('cost time:',end_time-start_time)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('E:\\AI_workspace\\pre_models\\chatglm3-6b',trust_remote_code=True)
    model = AutoModel.from_pretrained('E:\\AI_workspace\\pre_models\\chatglm3-6b',trust_remote_code=True)

    model = model.to(device)

    sentences = [
        "加拿大（英语/法语: Canada），首都渥太华，位于北美洲北部。东临大西洋，西濒太平洋，西北部邻美国阿拉斯加州，南接美国本土，北靠北冰洋。气候大部分为亚寒带针叶林气候和湿润大陆性气候，北部极地区域为极地长寒气候。",
        "《琅琊榜》是由山东影视传媒集团,山东影视制作有限公司, 北京儒意欣欣影业投资有限公司,北京和颂天地影视文化有限公司, 北京圣基影业有限公司, 东阳正午阳光影视有限公司联合出品, 由孔笙, 李雪执导, 胡歌, 刘涛, 王凯, 黄维德, 陈龙, 吴磊, 高鑫等主演的古装剧。",
        "《满江红》是由张艺谋执导, 沈腾, 易烊千玺, 张译, 雷佳音, 岳云鹏, 王佳怡领衔主演, 潘斌龙, 余皑磊主演, 郭京飞, 欧豪友情出演, 魏翔, 张弛, 黄炎特别出演, 许静雅, 蒋鹏宇, 林博洋, 飞凡, 任思诺, 陈永胜出演的悬疑喜剧电影。",
        "布宜诺斯艾利斯（Buenos Aires,华人常简称为布宜诺斯）是阿根廷共和国（the Republic of Argentina, Republica Argentina）的首都和最大城市，位于拉普拉塔河南岸，南美洲东南部，河对岸为乌拉圭东岸共和国。",
        "张译（原名张毅），1978年2月17日出生于黑龙江省哈尔滨市，中国内地演员。1997年至2006年服役于北京军区政治部战友话剧团。2006年，主演军事励志题材电视剧《士兵突击》。"
    ]

    pre_history = init_prompts()

    predict(sentences,pre_history)