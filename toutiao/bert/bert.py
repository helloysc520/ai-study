import os
import torch
from transformers import BertTokenizer, BertConfig, BertModel
import torch.nn as nn

class Config(object):

    def __init__(self,dataset):

        self.model_name = 'bert'
        self.data_path = 'E:\\AI_workspace\\ai-study\\toutiao\\data\\'
        self.train_path = self.data_path + 'train.txt'
        self.dev_path = self.data_path + 'dev.txt'
        self.test_path = self.data_path + 'test.txt'

        self.class_list = [x.strip() for x in open(self.data_path + 'class.txt').readlines()]

        self.save_path = 'E:\\AI_workspace\\ai-study\\toutiao\\bert\\'

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.save_path = self.save_path + self.model_name + '.pt' #模型训练结果

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 1000 #若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)
        self.num_epochs = 3
        self.batch_size = 128
        self.pad_size = 32 #每句话处理成的长度
        self.learning_rate = 5e-5
        self.bert_path = 'E:\\AI_workspace\\pre_models\\bert_pretrain\\'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_path + 'bert_config.json')
        self.hidden_size = self.bert_config.hidden_size


class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path,config=config.bert_config)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)


    def forward(self,x):

        content = x[0]
        mask = x[2]

        outputs = self.bert(content, attention_mask=mask)

        out = self.fc(outputs.pooler_output)

        return out



