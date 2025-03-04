from rouge import Rouge


import os
import sys
from predict import Predict


root_path = os.path.dirname(os.path.abspath(__file__))
util_path = os.path.join(root_path,'utils')
sys.path.append(util_path)

from utils.func_util import timer
from utils import config

class RougeEval():

    def __init__(self,path):

        self.path = path
        self.scores = None
        self.rouge = Rouge()
        self.sources = []
        self.hypos = []
        self.refs = []
        self.process()

    def process(self):

        print('Reading from ',self.path)
        with open(self.path,'r') as test:
            for line in test:
                source,ref = line.strip().split('<SEP>')
                ref = ref.replace('。','.')
                self.sources.append(source)
                self.refs.append(ref)

        print('self.refs[]包含的样本数:',len(self.refs))
        print(f'Test set contains {len(self.sources)} samples.')

    @timer('building hypotheses')
    def build_hypos(self):
        print('Building hypotheses.')
        count = 0
        for source in self.sources:
            count += 1
            if count % 1000  == 0:
                print('count=',count)
            self.hypes.append(predict.predict(source))


    def get_average(self):

        assert len(self.hypos) > 0, '需要首先构建hypotheses'
        print('calculating average rouge scores.')
        return self.rouge.get_scores(self.hypos,self.refs,avg=True)


if __name__ == '__main__':

    ##调用
    print('实例化Rouge对象...')
    rouge_eval = RougeEval(config.val_data_path)

    print('实例化Predict对象...')
    predict = Predict()

    #利用模型对article进行预测
    print('利用模型对article进行预测，并通过Rouge对象进行评估...')
    rouge_eval.build_hypos(predict)

    #将预测结果和标签abstract进行rouge规则计算
    result = rouge_eval.get_average()

    print('rouge1:',result['rouge-1'])
    print('rouge2:',result['rouge-2'])
    print('rouge3:',result['rouge-3'])

    print('将评估结果写入结果文件中...')
    with open('../eval_result/rouge_result.txt','a') as f:
        for r,metrics in result.items():
            f.write(r + '\n')
            for metric,value in metrics.items():
                f.write(metric + ':' + str(value * 100) + '\n')



