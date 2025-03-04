'''
采用分词的数据训练模型
'''

import fasttext
import time

train_data_path = 'E:\\AI_workspace\\ai-study\\toutiao\\data\\train_fenci_fast.txt'
test_data_path = 'E:\\AI_workspace\\ai-study\\toutiao\\data\\test_fenci_fast.txt'
dev_data_path = 'E:\\AI_workspace\\ai-study\\toutiao\\data\\dev_fenci_fast.txt'

'''
autotuneValidationFile 指定验证数据集合，它将在验证集上使用随机搜索方法寻找可能最优的超参数

使用autotuneDuration 参数可以控制随机搜索的时间，默认是300s，根据不同的需求，我们可以延长或缩短时间

verbose： 决定日志打印级别，当设置为3，可以将当前正在尝试的超参数打印出来

'''
model = fasttext.train_supervised(input=train_data_path,
                                  autotuneValidationFile=dev_data_path,
                                  dim= 64,
                                  autotuneDuration= 600,
                                  wordNgrams= 2)

#在测试集上评估模型的表现
result = model.test(test_data_path)
print(result)

#模型保存

time1 = int(time.time())

model_save_path = 'E:\\AI_workspace\\ai-class-code-demo\\toutiao\\baseline\\fast_text\\toutiao_fast_{}.bin'.format(time1)

model.save_model(model_save_path)