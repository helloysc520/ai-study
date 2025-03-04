import pickle
import numpy as np
import torch
import os
import sys

from torch.nn.utils import clip_grad_norm

from PGN import PGN
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from evaluate import evaluate


root_path = os.path.dirname(os.path.abspath(__file__))
util_path = os.path.join(root_path,'utils')
sys.path.append(util_path)

from utils import config
from utils.dataset import SampleDataset,collate_fn,PairDataset
from utils.func_util import config_info


#编写训练函数
def train(dataset,val_dataset,v,start_epoch=0):

    DEVICE = config.DEVICE

    model = PGN(v)
    model = model.to(DEVICE)

    print('loading data ......')
    train_data = SampleDataset(dataset.pairs,v)
    val_data = SampleDataset(val_dataset.pairs,v)

    print('initializing optimizer......')

    #定义模型训练的优化器
    optimizer = optim.Adam(model.parameters(),lr=config.learning_rate)

    #定义测试数据迭代器
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    #验证集上的损失值初始化为一个大整数
    val_losses = 10000000.0

    #summarywriter:为服务于Tensorboardx 写日志的可视化工具
    writer = SummaryWriter(config.log_path)

    num_epochs = len(range(start_epoch,config.num_epochs))

    #训练阶段采用Teacher-forcing的策略
    teacher_forcing = True
    print('teacher forcing = {}'.format(teacher_forcing))

    #根据配置文件config.py中的设置，对整个数据进行一轮的迭代训练
    with tqdm(total=config.epochs) as epoch_process:
        for epoch in range(start_epoch,config.epochs):
            #每一个epoch之前打印模型训练的相关配置信息
            print(config_info(config))

            batch_losses = []
            num_batches = len(train_dataloader)

            #针对每一个epoch，按batch读取数据迭代训练模型
            with tqdm(total=num_batches // 100) as batch_process:
                for batch,data in enumerate(tqdm(train_dataloader)):
                    x,y,x_len,y_len,oov,len_oov = data
                    assert not np.any(np.isnan(x.numpy()))

                    #如果配置GPU，则加速训练
                    if config.is_cuda:
                        x = x.to(DEVICE)
                        y = y.to(DEVICE)
                        x_len = x_len.to(DEVICE)
                        len_oov = len_oov.to(DEVICE)

                    #设置模型进入训练模式
                    model.train()

                    optimizer.zero_grad()

                    loss = model(x,x_len,y,len_oov,batch=batch,num_batches=num_batches,teacher_forcing=teacher_forcing)

                    batch_losses.append(loss.item())

                    loss.backward()

                    #为防止梯度爆炸
                    clip_grad_norm(model.encoder.parameters(), config.max_grad_norm)
                    clip_grad_norm(model.decoder.parameters(), config.max_grad_norm)
                    clip_grad_norm(model.attention.parameters(), config.max_grad_norm)

                    optimizer.step()

                    #每隔100个batch记录一下损失值信息
                    if (batch % 100) == 0:
                        batch_process.set_description(f'Epoch{epoch}')
                        batch_process.set_postfix(Batch=batch,Loss=loss.item())
                        batch_process.update()

                        #向tensorboard中写入损失值信息
                        writer.add_scalar(f'Average loss for epoch {epoch}', np.mean(batch_losses), global_step=batch)

                epoch_loss = np.mean(batch_losses)

                epoch_process.set_description(f'Epoch {epoch}')
                epoch_process.set_postfix(Loss=epoch_loss)
                epoch_process.update()

                #结束每一个epoch训练后，直接在验证集上跑一下模型效果
                avg_val_loss = evaluate(model,val_data,epoch)

                print('training loss:{}.'.format(epoch_loss),'validation loss:{}.'.format(avg_val_loss))

                #更新更小的验证集损失值evaluating loss
                if(avg_val_loss < val_losses):

                    torch.save(model.encoder,config.encoder_save_name)
                    torch.save(model.decoder,config.decoder_save_name)
                    torch.save(model.attention,config.attention_save_name)
                    torch.save(model.reduce_state,config.reduce_state_save_name)
                    torch.save(model.state_dict,config.model_save_path)
                    val_losses = avg_val_loss

                    #将更小的损失值写入文件中
                    with open(config.losses_path,'wb') as f:
                        pickle.dump(val_losses,f)



    writer.close()



if __name__ == '__main__':

    #构建训练用的数据集
    dataset = PairDataset(config.train_data_path,
                          max_enc_len=config.max_enc_len,
                          max_dec_len=config.max_dec_len,
                          truncate_enc=config.truncate_enc,
                          truncate_dec=config.truncate_dec)


    ##构建测试用的数据集
    val_dataset = PairDataset(config.val_data_path,
                          max_enc_len=config.max_enc_len,
                          max_dec_len=config.max_dec_len,
                          truncate_enc=config.truncate_enc,
                          truncate_dec=config.truncate_dec)

    #创建模型的单词词典
    vocab = dataset.build_vocab(embed_file=config.embed_file)

    train(dataset,val_dataset,vocab,start_epoch=0)






