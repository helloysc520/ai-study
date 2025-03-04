import os
import sys

import numpy as np
import time
import heapq
import random
import pathlib
import torch
import config


def timer(module):

    def wrapper(func):

        def cal_time(*args, **kwargs):

            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            cost_time = t2 - t1
            print(f'{cost_time} secs used for', module)
            return res

        return cal_time

    return wrapper

@timer(module= 'test a demo program')
def test():
    s = 0
    for i in range(1000000000):
        s += i
    print('s=',s)


def simple_tokenizer(text):

    return text.split()

def count_words(counter,text):

    for sentence in text:
        for word in sentence:
            counter[word] += 1

def sort_batch_by_len(data_batch):

    #初始化一个结果字典，其中包含的6个字段都是数据迭代器中的6个字段
    res = {
        'x': [],
        'y': [],
        'x_len':[],
        'y_len':[],
        'OOV':[],
        'len_OOV': []
    }

    for i in range(len(data_batch)):
        res['x'].append(data_batch[i]['x'])
        res['y'].append(data_batch[i]['y'])
        res['x_len'].append(len(data_batch[i]['x']))
        res['y_len'].append(len(data_batch[i]['y']))
        res['OOV'].append(data_batch[i]['OOV'])
        res['len_OOV'].append((data_batch[i]['len_OOV']))

    #以 x_len字段大小进行排序，并返回下标结果的列表
    sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()

    #返回的data_batch依然保持字典类型
    data_batch = {name : [_tensor[i] for i in sorted_indices] for name, _tensor in res.items()}

    return data_batch

#原始文本映射成ids
def source2ids(source_words,vocab):
    ids = []
    oovs = []
    unk_id = vocab.UNK
    for w in source_words:
        i = vocab[w]
        if i == unk_id:  #如果w是oov单词
            if w not in oovs: #将w添加到oov列表
                oovs.append(w)

            oov_num = oovs.index(w)
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)

    return ids,oovs

def abstract2ids(abstract_words,vocab,source_oovs):
    ids = []
    unk_id = vocab.UNK
    for w in abstract_words:
        i = vocab[w]
        if i == unk_id:
            if w  in source_oovs: ##如果w是source document oov ，则计算出一个新的映射id值
                vocab_idx = vocab.size() + source_oovs.index(w)
                ids.append(vocab_idx)

            else:  ##如果w不是source document oov,则只能用unk的id值代替
                ids.append(unk_id)
        else:
            ids.append(i)

    return ids

def outputids2words(id_list,source_oovs,vocab):

    words = []

    for i in id_list:

        try:
            w = vocab.index2word[i]
        except IndexError:
            assert_msg = 'Error: 无法在词典中找到该id值.'
            assert source_oovs is not None, assert_msg

            #希望索引i是一个source document oov单词
            source_oov_idx = i - vocab.size()
            try:
                w = source_oovs[source_oov_idx]
            except ValueError:
                raise ValueError('Error:模型生成的ID：%i,原始文本中的OOV ID:%i 但是当前样本中只有%i个oovs' % (i,source_oov_idx,source_oovs))

            words.append(w)

    return ' '.join(words)

#创建小顶堆
def add2heap(heap,item,k):

    if len(heap) < k:
       heapq.heappush(heap,item)
    else:
        heapq.heappushpop(heap,item)


def replace_oovs(in_tensor,vocab):

    oov_token = torch.full(in_tensor.shape,vocab.UNK,dtype=torch.long).to(config.DEVICE)
    out_tensor = torch.where(in_tensor > len(vocab) - 1 ,oov_token,in_tensor)
    return out_tensor

def config_info(config):

    info = 'model_name = {},pointer = {},coverage = {},fine_tune = {},scheduled_sampling ={},weight_tying = {},' + 'source = {}'
    return (info.format(config.model_name,config.pointer,config.coverage,config.fine_tune,config.scheduled_sampling,config.weight_tying,config.source))



if __name__ == '__main__':
    test()