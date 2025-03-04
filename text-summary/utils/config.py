import torch
import pgn_config


##神经网络参数
hidden_size = 512
dec_hidden_size = 512
embed_size = 512
pointer = True

#模型相关配置参数
max_vocab_size = 20000
model_name = 'pgn_model'
embed_file = pgn_config.word_vector_path
source = 'train'
train_data_path = pgn_config.train_data_path
test_data_path = pgn_config.test_data_path
val_data_path = pgn_config.dev_data_path
stop_word_file = pgn_config.stop_words_path
losses_path = pgn_config.loss_path
log_path = pgn_config.log_path
word_vector_model_path = pgn_config.word_vector_path
encoder_save_name = "E:\\AI_workspace\\saveModels\\text-summary项目\\pgn\\model_encoder.pt"
decoder_save_name = 'E:\\AI_workspace\\saveModels\\text-summary项目\\pgn\\model_decoder.pt'
attention_save_name = 'E:\\AI_workspace\\saveModels\\text-summary项目\\pgn\\model_attention.pt'
reduce_state_save_name = 'E:\\AI_workspace\\saveModels\\text-summary项目\\pgn\\model_reduce_state.pt'
model_save_path = 'E:\\AI_workspace\\saveModels\\text-summary项目\\pgn\\pgn_model.pt'
max_enc_len = 200
max_dec_len = 100
truncate_enc = True
truncate_dec = True

#下面两个参数关系到predict阶段的展示效果，
min_dec_steps = 30

#需要按业务场景调参,在greedy decode 设置 50,beam-search decode 设置30
max_dec_steps = 30
enc_rnn_dropout = 0.5
enc_attn = True
dec_attn = True
dec_in_dropout = 0
dec_rnn_dropout = 0
dec_out_dropout = 0

#训练参数
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 0.001
lr_decay = 0.0
initial_accumulator_value = 0.1
epochs = 2
num_epochs = 2
batch_size = 32
is_cuda = True

#下边4个参数都是优化策略
coverage = False
fine_tune = False
scheduled_sampling = False
weight_tying = False

max_grad_norm = 2.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LAMBDA = 1

