import torch
import torch.nn as nn
import os
import sys
from model import Attention,Encoder,Decoder,ReduceState

root_path = os.path.dirname(os.path.abspath(__file__))
util_path = os.path.join(root_path,'utils')
sys.path.append(util_path)

from utils import config
from utils.func_util import replace_oovs
from utils.vocab import Vocab


class PGN(nn.Module):

    def __init__(self,v):
        super(PGN,self).__init__()
        #初始化字典对象
        self.v = v
        self.DEVICE = config.DEVICE

        #依次初始化4个类对象
        self.attention = Attention(config.hidden_size)
        self.encoder = Encoder(len(v),config.embed_size,config.hidden_size)
        self.decoder = Decoder(len(v),config.embed_size,config.hidden_size)
        self.reduce_state = ReduceState()


    #计算最终分布
    def get_final_distribution(self,x,p_gen,p_vocab,attention_weights,max_oov):

        if not config.pointer:
            return p_vocab

        batch_size = x.size()[0]
        #进行p_gen概率值的裁剪,具体的取值范围可以参考
        p_gen = torch.clamp(p_gen,0.001,0.999)
        #接下来两行代码是论文公式9的计算
        p_vocab_weighted = p_gen * p_vocab
        #(batch_size,seq_len)
        attention_weighted = (1 - p_gen) * attention_weights

        #得到拓展后的单词概率分布（extended-vocab probability distribution）
        extension = torch.zeros((batch_size,max_oov)).float().to(self.DEVICE)
        #(batch_size,extended_vocab_size)
        p_vocab_extended = torch.cat([p_vocab_weighted,extension],dim=1)

        #根据论文中的公式9，累加注意力值
        final_distribution = p_vocab_extended.scatter_add_(dim=1,index=x,src=attention_weighted)

        return final_distribution

    def forward(self,x,x_len,y,len_oovs,batch,num_batches,teacher_forcing):

        x_copy = replace_oovs(x,self.v)
        x_padding_masks = torch.ne(x,0).byte().float()

        #第一步
        encoder_output,encoder_states = self.encoder(x_copy)
        decoder_states = self.reduce_state(encoder_states)

        #初始化每一步的损失值
        step_losses = []

        #第二步:循环解码,每一个时间步都经历注意力的计算，解码器的计算
        x_t = y[:,0]

        for t in range(y.shape[1] - 1):

            #如果使用Teacher_foring,则每一个时间步用真实标签来指导训练
            if teacher_forcing:
                x_t = y[:, t]

            x_t = replace_oovs(x_t,self.v)
            y_t = y[:,t+1]

            #通过注意力层计算context_vector
            context_vector,attention_weights = self.attention(decoder_states,encoder_output,x_padding_masks)

            #通过解码器计算得到的vocab distribution 和 hidden_states
            p_vocab,decoder_states,p_gen = self.decoder(x_t.unsqueeze(1),decoder_states,context_vector)

            #最终的概率分布
            final_dist = self.get_final_distribution(x,p_gen,p_vocab,attention_weights,torch.max(len_oovs))

            #第t个时间的预测结果，将作为t + 1个时间步的输入(如果采用Teacher-foring则不同)
            x_t = torch.argmax(final_dist,dim=1).to(self.DEVICE)

            #根据模型对target tokens的预测，来获取到猜测的概率
            if not config.pointer:
                y_t = replace_oovs(y_t,self.v)

            target_probs = torch.gather(final_dist,1,y_t.unsqueeze(1))
            target_probs = target_probs.squeeze(1)

            #将解码器端的PAD用padding mask 遮掩掉，防止计算loss的干扰
            mask = torch.ne(y_t,0).byte()
            #为防止计算log(0)而做的数学上的平滑处理
            loss = -torch.log(target_probs + config.eps)

            #先遮掩
            mask = mask.float()
            loss = loss * mask
            step_losses.append(loss)


        #第三步：计算一个批次样本的损失值，为反向传播做准备
        sample_losses = torch.sum(torch.stack(step_losses,1),1)

        #统计非PAD的字符个数，作为当前批次序列的有效长度
        seq_len_mask = torch.ne(y,0).byte().float()
        batch_seq_len = torch.sum(seq_len_mask,dim =1)

        #计算批次样本的平均损失值
        batch_loss = torch.mean(sample_losses/batch_seq_len)
        return batch_loss

if __name__ == '__main__':

    v = Vocab()
    model = PGN(v)
    print(model)

