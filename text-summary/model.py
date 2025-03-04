import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

root_path = os.path.dirname(os.path.abspath(__file__))
util_path = os.path.join(root_path,'utils')
sys.path.append(util_path)

from utils import  config



class Encoder(nn.Module):

    def __init__(self,vocab_size,embed_size,hidden_size,rnn_drop=0):

        super(Encoder,self).__init__()

        #词嵌入采用跟随模型一起训练的模式
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.hidden_size = hidden_size

        #编码器的主主体采用单层，双向lstm架构
        self.lstm = nn.LSTM(embed_size,hidden_size,bidirectional=True,dropout=rnn_drop,batch_first=True)

    def forward(self,x):

        embed = self.embedding(x)
        output,hidden = self.lstm(embed)
        return output,hidden

class Attention(nn.Module):

    def __init__(self,hidden_units):
        super(Attention,self).__init__()
        #定义前向传播层，对应论文中的公式1中的wh,ws
        self.wh = nn.Linear(2 * hidden_units,2 * hidden_units,bias=False)
        self.ws = nn.Linear(2 * hidden_units,2 * hidden_units)

        #定义全连接层，对应论文中的公式1中的最外层的v
        self.v = nn.Linear(2 * hidden_units,1,bias=False)


    def forward(self,decoder_states,encoder_output,x_padding_masks):
        h_dec,c_dec = decoder_states

        #将两个张量在最后一个维度拼接，得到decoder state st:(1,batch_size,2*hidden_units)
        s_t = torch.cat([h_dec,c_dec],dim=2)

        #将batch_size 置于第一个维度上，(batch_size,1,2 * hidden_units)
        s_t = s_t.transpose(0,1)

        #按照hi的维度拓展st的维度，(batch_size,seq_length,2*hidden_units)
        s_t = s_t.expand_as(encoder_output).contiguous()

        #根据论文中的公式1来计算et，总共有三步
        #第一步：分别经历各自的全连接层矩阵乘法
        #wh * h_i: (batch_size,seq_length,2 * hidden_units)
        encoder_features = self.wh(encoder_output.contiguous())
        #ws * s_t: (batch_size,seq_length,2 * hidden_units)
        decoder_features = self.ws(s_t)

        #第二步：两部分执行加和运算
        attn_inputs = encoder_features + decoder_features

        #第三步: 执行tanh运算和一个全连接层的运算
        #(batch_size,seq_length,1)
        score = self.v(torch.tanh(attn_inputs))

        #得到score后，执行论文中的公式2
        #(batch_size,seq_length)
        attention_weights = F.softmax(score,dim = 1).squeeze(2)

        #添加一步执行padding mask的运算，将编码器无效的PAD字符全部遮掩掉
        attention_weights = attention_weights * x_padding_masks

        #整个注意力层执行一次正则化操作
        normalization_factor = attention_weights.sum(1,keepdim=True)
        attention_weights = attention_weights / normalization_factor

        ##执行论文中的公式3，将上一步得到的attention distributon应用在encoder hidden state上,得到context_vector
        context_vector = torch.bmm(attention_weights.unsqueeze(1),encoder_output)

        #(batch_size,2 * hidden_units)
        context_vector = context_vector.squeeze(1)

        return context_vector,attention_weights

class Decoder(nn.Module):

    def __init__(self,vocab_size,embed_size,hidden_size,enc_hidden_size=None):

        super(Decoder,self).__init__()

        #解码器端也采用跟随模型一起训练的方式，得到词嵌入层
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        #解码器的主体结构采用单向lstm
        self.lstm = nn.LSTM(embed_size,hidden_size,batch_first=True)

        #因为要将decoder hidden state 和 context vector拼接,因此需要3倍的hidden_size
        self.w1 = nn.Linear(self.hidden_size * 3,self.hidden_size)
        self.w2 = nn.Linear(self.hidden_size,vocab_size)

        if config.pointer:
            #根据论文中的公式8进行运算，所谓输入维度上匹配的是4 * hidden_size + embed_size
            self.w_gen = nn.Linear(self.hidden_size * 4 + embed_size,1)

    def forward(self,x_t,decoder_states,context_vector):
        #首先计算Decoder的前向传播输出张量
        decoder_emb = self.embedding(x_t)
        decoder_output,decoder_states = self.lstm(decoder_emb,decoder_states)

        #接下来就是论文中公式4的计算
        decoder_output = decoder_output.view(-1,config.hidden_size)
        concat_vector = torch.cat([decoder_output,context_vector],dim = -1)

        #经历两个全连接层v和v1后，再进行softmax计算，得到vocabbulary distribution
        #(batch_size,hidden_units)
        FF1_out = self.w1(concat_vector)

        #(batch_size,vocab_size)
        FF2_out = self.w2(FF1_out)

        #(batch_size,vocab_size)
        p_vocab = F.softmax(FF2_out,dim=-1)

        #构造decoder state s_t
        h_dec,c_dec = decoder_states

        #(1,batch_size,2 * hidden_units)
        s_t = torch.cat([h_dec,c_dec],dim=2)

        #p_gen 是通过context vector h_t decoder state s_t ,decoder input x_t,三个部分共同计算出来的
        p_gen = None

        if config.pointer:
            #这里采用了直接拼接3部分输入张量，然后经历一个共同的全连接层w_gen,和原始论文的计算不同
            x_gen = torch.cat([context_vector,s_t.squeeze(0),decoder_emb.squeeze(1)],dim= -1)
            p_gen = torch.sigmoid(self.w_gen(x_gen))

        return p_vocab,decoder_states,p_gen


class ReduceState(nn.Module):

    def __init__(self):
        super(ReduceState,self).__init__()

    def forward(self,hidden):

        h,c = hidden
        h_reduced = torch.sum(h,dim=0,keepdim=True)
        c_reduced = torch.sum(c,dim=0,keepdim=True)
        return (h_reduced,c_reduced)