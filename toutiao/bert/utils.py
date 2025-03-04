from datetime import time, timedelta

from tqdm import tqdm
import torch
import time

def build_vocab(file_path,tokenizer,max_size,min_freq):

    vocab_dic = {}
    with open(file_path,'r',encoding='utf-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue

            content = lin.split('\t')[0]

            for word in tokenizer(content):

                vocab_dic[word] = vocab_dic.get(word,0) + 1

        vocab_list = sorted(
            [_ for _ in vocab_dic.items() if _[1] >= min_freq],key=lambda x: x[1],reverse=True)[:max_size]

        vocab_dic = {word_count[0]:idx for idx, word_count in enumerate(vocab_list)}

        vocab_dic.update({'[UNK]':len(vocab_dic),'[PAD]':len(vocab_dic) + 1})

        return vocab_dic



def build_dataset(config):

    def load_dataset(path,pad_size=32):

        contents = []
        with open(path,'r',encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue

                content,label = line.split('\t')
                token = config.tokenizer.tokenize(content)
                token =  ['CLS'] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:

                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += [0] * (pad_size - len(token))


                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size

                contents.append((token_ids,int(label),seq_len,mask))

        return contents


    train = load_dataset(config.train_path,config.pad_size)
    dev = load_dataset(config.dev_path,config.pad_size)
    test = load_dataset(config.test_path,config.pad_size)

    return train,dev,test


"""
数据迭代器
"""
class DatasetIterator(object):

    def __init__(self,batches,batch_size,device,model_name):

        self.batches = batches
        self.batch_size = batch_size
        self.model_name = model_name
        self.n_batches = len(batches) // batch_size
        self.residue = False  #记录batch的数量是不是整数

        if len(batches) % self.n_batches != 0:
            self.residue = True

        self.device = device
        self.index = 0

    def _to_tensor(self,datas):

        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)

        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        if self.model_name == 'bert' or self.model_name == 'multi_mask_bert':
            mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)

            return (x,seq_len,mask),y

    def __iter__(self):
        return self

    def __next__(self):

        if self.residue and self.index == self.n_batches:

            batches = self.batches[self.index * self.batch_size : len(self.batches)]

            self.index += 1

            batches = self._to_tensor(batches)

            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration

        else:

            batches = self.batches[self.index * self.batch_size : (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches


    def __len__(self):

        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches




def build_iterator(dataset,config):

    iter = DatasetIterator(dataset,config.batch_size,config.device,config.model_name)

    return iter

def get_time_dif(start_time):

    end_time = time.time()
    time_diff = end_time - start_time

    return timedelta(seconds=int(round(time_diff)))