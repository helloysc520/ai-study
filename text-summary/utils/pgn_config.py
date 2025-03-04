import os


root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#原始数据文本路径
train_raw_data_path = os.path.join(root_path, "data", "train.csv")
test_raw_data_path = os.path.join(root_path, "data", "test.csv")

#停用词表和用户自定义字典的存储路径
stop_words_path = os.path.join(root_path, "data", "stopwords.txt")
user_dict_path = os.path.join(root_path, "data", "user_dict.txt")

#预处理 + 切分后的训练数据和测试数据路径
train_seg_path = os.path.join(root_path, "data", "pgn_train_seg_data.csv")
test_seg_path = os.path.join(root_path, "data", "pgn_test_seg_data.csv")


#经过第一轮处理后的最终训练数据和测试数据
train_data_path = os.path.join(root_path, "data", "pgn_train_2.txt")
test_data_path = os.path.join(root_path, "data", "pgn_test.txt")
dev_data_path = os.path.join(root_path, "data", "pgn_dev.txt")

#词向量模型路径
word_vector_path = os.path.join(root_path, "data",'wv', "word2vec.model")

loss_path = os.path.join(root_path, "data", "pgn_loss.txt")
log_path = os.path.join(root_path, "data", "pgn_log.txt")


