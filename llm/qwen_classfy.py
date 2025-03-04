import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

class_examples = {
        '人物': '岳云鹏，本名岳龙刚，1985年4月15日出生于河南省濮阳市南乐县，中国内地相声、影视男演员 [1] 。2005年，首次登台演出。2012年，主演卢卫国执导的喜剧电影《就是闹着玩的》。2013年在北京举办相声 专场。',
        '书籍': '《三体》是刘慈欣创作的长篇科幻小说系列，由《三体》《三体2:黑暗森林》《三体3:死神永 生》组成，第一部于2006年5月起在《科幻世界》杂志上连载，第二部于2008年5月首次出版，第三部则于2010年11 月出版。',
        '电视剧': '《狂飙》是由中央电视台、爱奇艺出品，留白影视、中国长安出版传媒联合出品，中央政法委 宣传教育局、中央政法委政法综治信息中心指导拍摄，徐纪周执导，张译、张颂文、李一桐、张志坚、吴刚领衔主演， 倪大红、韩童生、李建义、石兆琪特邀主演，李健、高叶、王骁等主演的反黑刑侦剧。',
        '电影': '《流浪地球》是由郭帆执导，吴京特别出演、屈楚萧、赵今麦、李光洁、吴孟达等领衔主演的科 幻冒险电影。影片根据刘慈欣的同名小说改编，故事背景设定在2075年，讲述了太阳即将毁灭，毁灭之后的太阳系已 经不适合人类生存，而面对绝境，人类将开启“流浪地球”计划，试图带着地球一起逃离太阳系，寻找人类新家园的故 事。',
        '城市': '乐山，古称嘉州，四川省辖地级市，位于四川中部，四川盆地西南部，地势西南高，东北低，属 中亚热带气候带;辖4区、6县，代管1个县级市，全市总面积12720.03平方公里;截至2021年底，全市常住人口 315.1万人。',
        '国家': '瑞士联邦(Swiss Confederation)，简称“瑞士”，首都伯尔尼，位于欧洲中部，北与德国 接壤，东临奥地利和列支敦士登，南临意大利，西临法国。地处北温带，四季分明，全国地势高峻，矿产资源匮乏，森 林及水力资源丰富，总面积41284平方千米，全国由26个州组成(其中6个州为半州)。'
    }

def init_prompts():

    class_list = list(class_examples.keys())

    pre_history = [

        {
            'content':f'你是一个文本分类专家，请将输入的文本分类到：{class_list}类别中.',
            'role': 'system'
        }
    ]

    for type,example in class_examples.items():

        new_data = {

            'content': f'{example}这段文本属于{type}类别',
            'role': 'user'
        }
        pre_history.append(new_data)

    return pre_history

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 导入 Qwen 大模型
    tokenizer = AutoTokenizer.from_pretrained('Qwen/qwen-7b-chat', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('Qwen/qwen-7b-chat', trust_remote_code=True)

    model = model.to(device)

    sentences = [
        "加拿大（英语/法语: Canada），首都渥太华，位于北美洲北部。东临大西洋，西濒太平洋，西北部邻美国阿拉斯加州，南接美国本土，北靠北冰洋。气候大部分为亚寒带针叶林气候和湿润大陆性气候，北部极地区域为极地长寒气候。",
        "《琅琊榜》是由山东影视传媒集团,山东影视制作有限公司, 北京儒意欣欣影业投资有限公司,北京和颂天地影视文化有限公司, 北京圣基影业有限公司, 东阳正午阳光影视有限公司联合出品, 由孔笙, 李雪执导, 胡歌, 刘涛, 王凯, 黄维德, 陈龙, 吴磊, 高鑫等主演的古装剧。",
        "《满江红》是由张艺谋执导, 沈腾, 易烊千玺, 张译, 雷佳音, 岳云鹏, 王佳怡领衔主演, 潘斌龙, 余皑磊主演, 郭京飞, 欧豪友情出演, 魏翔, 张弛, 黄炎特别出演, 许静雅, 蒋鹏宇, 林博洋, 飞凡, 任思诺, 陈永胜出演的悬疑喜剧电影。",
        "布宜诺斯艾利斯（Buenos Aires,华人常简称为布宜诺斯）是阿根廷共和国（the Republic of Argentina, Republica Argentina）的首都和最大城市，位于拉普拉塔河南岸，南美洲东南部，河对岸为乌拉圭东岸共和国。",
        "张译（原名张毅），1978年2月17日出生于黑龙江省哈尔滨市，中国内地演员。1997年至2006年服役于北京军区政治部战友话剧团。2006年，主演军事励志题材电视剧《士兵突击》。"
    ]

    history_prompts = init_prompts()

    model.eval()

    with torch.no_grad():
        for sentence in sentences:
            print('sentence:', sentence)
            start_time = time.time()

            # 每次将待预测的文本 sentence 和历史热身信息 history_prompts 送入大模型
            input_ids = tokenizer(sentence, return_tensors='pt').input_ids.to(device)
            output = model.generate(input_ids, max_length=100, num_return_sequences=1)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            end_time = time.time()

            print('response:', response)
            print('cost time:', end_time - start_time)
            print('*' * 100)