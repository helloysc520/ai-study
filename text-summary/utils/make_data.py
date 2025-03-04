import os
from pgn_config import *

#打开最终的结果文件

train_writer = open(train_data_path, 'w',encoding='utf-8')
test_writer = open(test_data_path, 'w',encoding='utf-8')

n = 0
with open(train_seg_path, 'r', encoding='utf-8') as f1:
    next(f1)
    for line in f1:
        line = line.strip().strip('\n')
        article,abstract = line.split('<sep>')
        text = abstract + '<SEP>' + article + '\n'
        train_writer.write(text)
        n += 1

print('train n=',n)

n = 0
with open(test_seg_path, 'r', encoding='utf-8') as f2:

    next(f2)
    for line in f2:
        line = line.strip().strip('\n')
        text = line + '\n'
        test_writer.write(text)
        n += 1

print('test n=',n)

