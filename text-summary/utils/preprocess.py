import re
import jieba
import pandas as pd
import numpy as np
import os
import sys


from pgn_config import *
from multi_proc_util import *

#jieba载入自定义切词表
jieba.load_userdict(user_dict_path)

#根据max_len 和 vocab 填充<START> <STOP> <PAD> <UNK>
def pad_proc(sentence,max_len,word_to_id):

    #1 按空格统计切分出词
    words = sentence.trip().split(' ')

    #2 截取最大长度的词
    words = words[:max_len]

    #3 填充UNK
    sentence = [w if w in word_to_id else '<UNK>' for w in words]

    #4 填充<START> 和 <STOP>
    sentence = ['<START>'] + sentence + ['<STOP>']

    #5 判断长度，填充 <PAD>
    sentence = sentence + ['<PAD>'] * (max_len - len(words))

    return ' '.join(sentence)

#加载停用词
def load_stop_words(stop_word_path):

    f = open(stop_word_path,'r',encoding='utf-8')

    stop_words = f.readlines()

    stop_words = [stop_word.strip() for stop_word in stop_words]

    return stop_words

##清洗文本
def clean_sentence(sentence):

    if isinstance(sentence,str):

        #删除1.2.3.  这些标题
        r = re.compile('\D(\d\.)\D')
        sentence = r.sub('',sentence)

        #删除带括号的 进口 海外
        r = re.compile(r'[((]进口[))]|\(海外\)')
        sentence = r.sub('',sentence)

        #删除除了汉字数字字母和,! ? 。 . - 以外的字符
        r = re.compile("[^，!?。\.\-\u4e00-\u9fa5_a-zA-z0-9]")
        #用中文输入法下的，！ ？ 来替换英文输入法下的, !,?
        sentence = sentence.replace(',','，')
        sentence = sentence.replace('!','！')
        sentence = sentence.replace('?','？')
        sentence = r.sub('', sentence)


        #删除 车主说 技师说 语音 图片 你好 您好
        r = re.compile(r'车主说|技师说|语音|图片|你好|您好')
        sentence = r.sub('',sentence)

        return sentence
    else:
        return ''


def filter_stopwords(seg_list):

    #去掉多余空字符
    words = [word for word in seg_list if word]

    # 去掉停用词
    return [word for word in words if word not in load_stop_words(stop_words_path)]

def sentence_proc(sentence):

    sentence = clean_sentence(sentence)
    words = jieba.cut(sentence)
    words = filter_stopwords(words)
    return ' '.join(words)

def sentences_proc(df):

    for col_name in ['Brand','Model','Question','Dialogue']:
        df[col_name] = df[col_name].apply(sentence_proc)

    if 'Report' in df.columns:
        df['Report'] = df['Report'].apply(sentence_proc)

    return df

def build_dataset(train_raw_data_path,test_raw_data_path):

    ##1.加载原始数据
    print('1.加载原始数据')
    print(train_raw_data_path)

    train_df = pd.read_csv(train_raw_data_path,engine='python',encoding='utf-8')
    test_df = pd.read_csv(test_raw_data_path,engine='python',encoding='utf-8')

    print('原始训练集行数{},测试集行数{}'.format(len(train_df),len(test_df)))
    print('\n')

    ##2.空值去除
    print('2.空值去除')
    train_df.dropna(subset=['Question' , 'Dialogue','Report'],how='any',inplace=True)
    test_df.dropna(subset=['Question', 'Dialogue'],how='any',inplace=True)
    print('空值去除后训练集行数{},测试集行数{}'.format(len(train_df),len(test_df)))

    ##3.多线程，批量数据预处理
    print('3.多线程，批量数据预处理')
    train_df = parallelize(train_df,sentences_proc)
    test_df = parallelize(test_df,sentences_proc)
    print('\n')
    print('senteces_proc has done!')

    ##4.合并训练测试集
    print('4.合并训练测试集,用于训练词向量')
    train_df['x'] = train_df[['Question','Dialogue']].apply(lambda x : ' '.join(x),axis=1)
    train_df['y'] = train_df[['Report']]

    #新建一行，按行堆积
    test_df['x']  = test_df[['Question','Dialogue']].apply(lambda x : ' '.join(x),axis=1)

    #5.保存分割处理好的train_seg_data.csv test_seg_data.csv
    print('5.保存分割处理好的train_seg_data.csv test_seg_data.csv')
    train_df = train_df.drop(['Question'],axis=1)
    train_df = train_df.drop(['Dialogue'],axis=1)
    train_df = train_df.drop(['Brand'],axis=1)
    train_df = train_df.drop(['Model'],axis=1)
    train_df = train_df.drop(['Report'],axis=1)
    train_df = train_df.drop(['QID'],axis=1)

    test_df = test_df.drop(['Question'],axis=1)
    test_df = test_df.drop(['Dialogue'],axis=1)
    test_df = test_df.drop(['Brand'],axis=1)
    test_df = test_df.drop(['Model'],axis=1)
    test_df = test_df.drop(['QID'],axis=1)
    test_df.to_csv(test_seg_path,index=None,header=True)

    train_df['data'] = train_df[['x','y']].apply(lambda x : '<sep>'.join(x),axis=1)
    train_df = train_df.drop(['x'],axis=1)
    train_df = train_df.drop(['y'],axis=1)
    train_df.to_csv(train_seg_path,index=None,header=True)
    print('The csv_file has saved')
    print('\n')

    print('6.后续是将步骤5的结果文件适当处理，保存为.txt文件')
    print('本程序代码所有工作执行完毕')

if __name__ == '__main__':

    build_dataset(train_raw_data_path,test_raw_data_path)