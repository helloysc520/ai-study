from collections import Counter
import numpy as np
import torch
import torch.nn as nn

from pgn_config import word_vector_path
from gensim.models import word2vec

class Vocab(object):

    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self):
        self.word2index = {}
        self.word2count = Counter()
        self.reserved = ['<PAD>','<SOS>','<EOS>','<UNK>']
        self.index2word = self.reserved[:]
        self.embedding_matrix = None

    #类词典中增加单词
    def add_words(self, words):

        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)

        self.word2count.update(words)


    #如果提前训练好词向量，则执行类内函数对embedding_matrix 赋值
    def load_embeddings(self, word_vector_path):

        wv_model = word2vec.Word2Vec.load(word_vector_path)
        self.embedding_matrix = wv_model.wv.vectors

    def __getitem__(self, item):

        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    def __len__(self):

        return len(self.index2word)

    def size(self):

        return len(self.index2word)

if __name__ == '__main__':

    vocab = Vocab()
    print(vocab)
    print('*' * 100)
    print(vocab.size())
    print('*' * 100)
    print(vocab.embedding_matrix)