import pandas as pd
import numpy as np
from collections import Counter
import jieba

content = pd.read_csv('E:\\AI_workspace\\ai-study\\toutiao\\data\\dev.txt',sep='\t',names=['sentence','label'])

print(content.head(10))

print(len(content))

count = Counter(content.label.values)

print(count)
print(len(count))
print('*' * 100)


'''
第二步：分析样本分布
'''

total = 0
for i,v in count.items():
    total += v

print(total)

for i,v in count.items():
    print(i,v / total * 100,'%')

print('*' * 100)

content['sentence_len'] = content['sentence'].apply(lambda x: len(x))

print(content.head(10))

length_mean = np.mean(content['sentence_len'])
length_std = np.std(content.sentence_len)
print('length_mean = ',length_mean)
print('length_std = ',length_std)

print('*' * 100)

'''
第三步： 进行分词处理
'''

def cut_sentence(s):
    return list(jieba.cut(s))


content['words'] = content['sentence'].apply(cut_sentence)
print(content.head(10))

content['words'] = content['sentence'].apply(lambda x: ' '.join(cut_sentence(x)))

content['words'] = content['words'].apply(lambda x: ' '.join(x.split())[:30])

content.to_csv('E:\\AI_workspace\\ai-class-code-demo\\toutiao\\data\\dev_new.csv')