from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch
import os
import sys

root_path = os.path.dirname(os.path.abspath(__file__))
util_path = os.path.join(root_path,'utils')
sys.path.append(util_path)

from utils import  config
from utils.dataset import collate_fn

def evaluate(model,val_data,epoch):

    print("validating")
    val_loss = []

    #评估模型需要参数不变
    with torch.no_grad():
        DEVICE = config.DEVICE
        #创建数据迭代器
        val_dataloader = DataLoader(
            dataset = val_data,
            batch_size = config.batch_size,
            shuffle = True,
            pin_memory= True,
            drop_last= True,
            collate_fn= collate_fn
        )

    for batch,data in enumerate(tqdm(val_dataloader)):
        x,y,x_len,y_len,oov,len_oovs = data
        if config.is_cuda:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            x_len = x_len.to(DEVICE)
            len_oovs = len_oovs.to(DEVICE)
        total_num = len(val_dataloader)

        loss = model(x,x_len,y,len_oovs,batch=batch,num_batches=total_num,teacher_forcing= True)

        val_loss.append(loss.item())

    return np.mean(val_loss)
